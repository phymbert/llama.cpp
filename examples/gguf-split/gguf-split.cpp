#include "llama.h"
#include "common.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <vector>

#include <stdio.h>
#include <string.h>
#include <climits>
#include <stdexcept>

#if defined(_WIN32)
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif

enum split_operation : uint8_t {
    SPLIT_OP_SPLIT,
    SPLIT_OP_MERGE,
};

struct split_params {
    std::string executable;
    split_operation operation = SPLIT_OP_SPLIT;
    int n_split_tensors = 128;
    bool no_tensors_in_metadata = false;
    std::string split_size;
    std::string input;
    std::string output;
};

static void split_print_usage(const char * executable) {
    const split_params default_params;
    printf("\n");
    printf("usage: %s [options] GGUF_IN GGUF_OUT\n", executable);
    printf("\n");
    printf("Apply a GGUF operation on IN to OUT.");
    printf("\n");
    printf("options:\n");
    printf("  -h, --help               show this help message and exit\n");
    printf("  --version                show version and build info\n");
    printf("  --split                  split GGUF to multiple GGUF (default)\n");
    printf("  --split-max-tensors N    max tensors in each split: default(%d)\n", default_params.n_split_tensors);
    printf("  --split-max-size N(G|M)  max size of each split: default unused. This is a soft limit.\n");
    printf("  --no-tensor-in-metadata  the first shard will not contain tensors data but only metadata, default %s.\n", default_params.no_tensors_in_metadata ? "disabled" : "enabled");
    printf("  --merge                  merge multiple GGUF to a single GGUF\n");
    printf("\n");
}

static bool split_params_parse_ex(int argc, const char ** argv, split_params & params) {
    std::string arg;
    const std::string arg_prefix = "--";
    bool invalid_param = false;

    int arg_idx = 1;
    for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++) {
        arg = argv[arg_idx];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        bool arg_found = false;
        if (arg == "-h" || arg == "--help") {
            split_print_usage(argv[0]);
            exit(0);
        }
        if (arg == "--version") {
            fprintf(stderr, "version: %d (%s)\n", LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
            fprintf(stderr, "built with %s for %s\n", LLAMA_COMPILER, LLAMA_BUILD_TARGET);
            exit(0);
        }

        if (arg == "--merge") {
            arg_found = true;
            params.operation = SPLIT_OP_MERGE;
        }
        if (arg == "--split") {
            arg_found = true;
            params.operation = SPLIT_OP_SPLIT;
        }
        if (arg == "--split-max-tensors") {
            if (++arg_idx >= argc) {
                invalid_param = true;
                break;
            }
            arg_found = true;
            params.n_split_tensors = atoi(argv[arg_idx]);
        }
        if (arg == "--split-max-size") {
            if (++arg_idx >= argc) {
                invalid_param = true;
                break;
            }
            arg_found = true;
            params.split_size = argv[arg_idx];
        }
        if (arg == "--no-tensor-in-metadata") {
            arg_found = true;
            params.no_tensors_in_metadata = true;
        }

        if (!arg_found) {
            throw std::invalid_argument("error: unknown argument: " + arg);
        }
    }

    if (invalid_param) {
        throw std::invalid_argument("error: invalid parameter for argument: " + arg);
    }

    if (argc - arg_idx < 2) {
        printf("%s: bad arguments\n", argv[0]);
        return false;
    }

    params.input = argv[arg_idx++];
    params.output = argv[arg_idx++];

    return true;
}

static bool split_params_parse(int argc, const char ** argv, split_params & params) {
    bool result = true;
    try {
        if (!split_params_parse_ex(argc, argv, params)) {
            split_print_usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        split_print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return result;
}

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

struct split_strategy {
    const split_params params;
    std::ifstream & f_input;
    struct gguf_context * ctx_gguf;
    struct ggml_context * ctx_meta = NULL;
    const int n_tensors; // total tensors to split

    int n_split = 0; // total split to generate

    int i_split = 0; // current split file index
    int i_split_tensors_data = 0; // current file index where to write the tensors data into

    int i_tensor = 0; // current tensor to write

    size_t n_bytes_written = 0;
    size_t n_total_bytes = 0;

    std::vector<uint8_t> read_data;

    struct gguf_context * ctx_out; // current split
    std::ofstream fout;

    split_strategy(const split_params & params,
            std::ifstream & f_input,
            struct gguf_context * ctx_gguf,
            struct ggml_context * ctx_meta) :
        params(params),
        f_input(f_input),
        ctx_gguf(ctx_gguf),
        ctx_meta(ctx_meta),
        n_tensors(gguf_get_n_tensors(ctx_gguf)) {
            for (ggml_tensor * cur = ggml_get_first_tensor(ctx_meta); cur; cur = ggml_get_next_tensor(ctx_meta, cur)) {
                n_total_bytes += ggml_nbytes(cur);
            }
        }

    virtual ~split_strategy() {}

    virtual bool should_split() const{
        return true;
    }

    virtual bool include_tensor(int /*idx*/, ggml_tensor * /*t*/) {
        return true;
    }

    virtual void split_start() {
        ctx_out = gguf_init_empty();

        // Save all metadata in first split only
        if (i_split == 0) {
            gguf_set_kv(ctx_out, ctx_gguf);
        }
        gguf_set_val_u16(ctx_out, LLM_KV_SPLIT_NO, i_split);
        gguf_set_val_u16(ctx_out, LLM_KV_SPLIT_COUNT, n_split);
        gguf_set_val_i32(ctx_out, LLM_KV_SPLIT_TENSORS_COUNT, n_tensors);

        // populate the split metadata
        i_split_tensors_data = params.no_tensors_in_metadata ? i_split - 1 : i_split;
        if (!params.no_tensors_in_metadata || i_split > 0) {
            for (int i = i_tensor; i < n_tensors; ++i) {
                struct ggml_tensor * meta = ggml_get_tensor(ctx_meta, gguf_get_tensor_name(ctx_gguf, i));
                if (!include_tensor(i, meta)) {
                    break;
                }
                gguf_add_tensor(ctx_out, meta);
            }
        }

        char split_path[PATH_MAX] = {0};
        llama_split_path(split_path, sizeof(split_path), params.output.c_str(), i_split, n_split);

        fprintf(stderr, "%s: %s ...", __func__, split_path);
        fout = std::ofstream(split_path, std::ios::binary);
        fout.exceptions(std::ofstream::failbit); // fail fast on write errors

        auto meta_size = gguf_get_meta_size(ctx_out);

        // placeholder for the metadata
        ::zeros(fout, meta_size);

        i_split++;
    }

    virtual void next_tensor() {
        const char * t_name = gguf_get_tensor_name(ctx_gguf, i_tensor);
        struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);
        size_t n_bytes = ggml_nbytes(t);

        if (read_data.size() < n_bytes) {
            read_data.resize(n_bytes);
        }

        auto offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i_tensor);
        f_input.seekg(offset);
        f_input.read((char *)read_data.data(), n_bytes);

        t->data = read_data.data();

        // write tensor data + padding
        fout.write((const char *)t->data, n_bytes);
        zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);

        i_tensor++;
        n_bytes_written += n_bytes;
    }

    void split_end() {
        // go back to beginning of file and write the updated metadata
        fout.seekp(0);
        std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
        gguf_get_meta_data(ctx_out, data.data());
        fout.write((const char *)data.data(), data.size());

        fout.close();
        gguf_free(ctx_out);

        fprintf(stderr, "\033[3Ddone\n");
    }
};

struct split_strategy_max_tensors : public split_strategy {

    split_strategy_max_tensors(const split_params & params,
                   std::ifstream & f_input,
                   struct gguf_context * ctx_gguf,
                   struct ggml_context * ctx_meta) :
        split_strategy(params, f_input, ctx_gguf, ctx_meta) {

        n_split = std::ceil(1. * n_tensors / params.n_split_tensors);
    }

    bool include_tensor(int idx, ggml_tensor * /*t*/) {
        return idx < (i_split_tensors_data + 1) * params.n_split_tensors;
    }

    bool should_split() const {
        return i_tensor % params.n_split_tensors == 0;
    }
};

struct split_strategy_max_size : public split_strategy {
    size_t n_bytes_to_write = 0;
    size_t n_bytes_per_split = 0;

    split_strategy_max_size(const split_params & params,
                   std::ifstream & f_input,
                   struct gguf_context * ctx_gguf,
                   struct ggml_context * ctx_meta) :
        split_strategy(params, f_input, ctx_gguf, ctx_meta) {
        int value = std::stoi(params.split_size.substr(0, params.split_size.size()-1));
        char unit = params.split_size[params.split_size.size()-1];

        switch (unit) {
            case 'M':
                n_bytes_per_split = value * 1024 * 1024;
                break;
            case 'G':
                n_bytes_per_split = value * 1024 * 1024 * 1024;
                break;
            default:
                fprintf(stderr, "%s: invalid max size %s\n", __func__, params.split_size.c_str());
                split_print_usage(params.executable.c_str());
                exit(EXIT_FAILURE);
        }

        n_split = std::min((int) std::ceil(1. * n_total_bytes / n_bytes_per_split), n_tensors);
    }

    void split_start() {
        n_bytes_to_write = 0;
        split_strategy::split_start();
    }

    bool include_tensor(int idx, ggml_tensor * t) {
        size_t n_bytes = ggml_nbytes(t);
        size_t max_bytes_to_write = (i_split_tensors_data + 1) * n_bytes_per_split;
        bool include = n_bytes_to_write + n_bytes <= max_bytes_to_write;
        if (!include && idx == i_tensor) {
            fprintf(stderr, "%s: ERROR --split-max-size too small for tensor %d: %zu > %zu\n", __func__, idx, n_bytes_to_write + n_bytes, max_bytes_to_write);
            exit(EXIT_FAILURE);
        }
        n_bytes_to_write += n_bytes;
        return include;
    }

    bool should_split() const {
        return n_bytes_written > (i_split_tensors_data + 1) * n_bytes_per_split;
    }
};

static void gguf_split(const split_params & split_params) {
    struct ggml_context * ctx_meta = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_meta,
    };

    std::ifstream f_input(split_params.input.c_str(), std::ios::binary);
    if (!f_input.is_open()) {
        fprintf(stderr, "%s:  failed to open input GGUF from %s\n", __func__, split_params.input.c_str());
        exit(EXIT_FAILURE);
    }

    auto * ctx_gguf = gguf_init_from_file(split_params.input.c_str(), params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s:  failed to load input GGUF from %s\n", __func__, split_params.input.c_str());
        exit(EXIT_FAILURE);
    }

    std::unique_ptr<split_strategy> strategy;
    if (!split_params.split_size.empty()) {
        strategy.reset(new split_strategy_max_size(split_params, f_input, ctx_gguf, ctx_meta));
    } else if (split_params.n_split_tensors > 0) {
        strategy.reset(new split_strategy_max_tensors(split_params, f_input, ctx_gguf, ctx_meta));
    } else {
        split_print_usage(split_params.executable.c_str());
        exit(EXIT_FAILURE);
    }

    if (split_params.no_tensors_in_metadata) {
        fprintf(stderr, "%s: first shard will only contain metadata\n", __func__);
        strategy->n_split++;
    }

    char first_split_path[PATH_MAX] = {0};
    llama_split_path(first_split_path, sizeof(first_split_path),
                     split_params.output.c_str(), strategy->i_split, strategy->n_split);
    fprintf(stderr, "%s: %s -> %s (%d tensors per file)\n",
            __func__, split_params.input.c_str(),
            first_split_path,
            split_params.n_split_tensors);

    strategy->split_start();

    if (split_params.no_tensors_in_metadata) {
        strategy->split_end();
        strategy->split_start();
    }

    while (strategy->i_tensor < strategy->n_tensors) {
        strategy->next_tensor();
        bool should_split = strategy->i_tensor < strategy->n_tensors && strategy->should_split();
        if (should_split) {
            strategy->split_end();
            strategy->split_start();
        }
    }
    strategy->split_end();

    gguf_free(ctx_gguf);
    f_input.close();

    fprintf(stderr, "%s: %d GGUF shards written with a total of %d tensors.\n",
            __func__, strategy->n_split, strategy->n_tensors);
}

static void gguf_merge(const split_params & split_params) {
    fprintf(stderr, "%s: %s -> %s\n",
            __func__, split_params.input.c_str(),
            split_params.output.c_str());
    int n_split = 1;
    int total_tensors = 0;

    auto * ctx_out = gguf_init_empty();
    std::ofstream fout(split_params.output.c_str(), std::ios::binary);
    fout.exceptions(std::ofstream::failbit); // fail fast on write errors

    std::vector<uint8_t> read_data;
    std::vector<ggml_context *> ctx_metas;
    std::vector<gguf_context *> ctx_ggufs;

    char split_path[PATH_MAX] = {0};
    strncpy(split_path, split_params.input.c_str(), sizeof(split_path) - 1);
    char split_prefix[PATH_MAX] = {0};

    // First pass to find KV and tensors metadata
    for (int i_split = 0; i_split < n_split; i_split++) {
        struct ggml_context * ctx_meta = NULL;

        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx_meta,
        };

        if (i_split > 0) {
            llama_split_path(split_path, sizeof(split_path), split_prefix, i_split, n_split);
        }
        fprintf(stderr, "%s: reading metadata %s ...", __func__, split_path);

        auto * ctx_gguf = gguf_init_from_file(split_path, params);
        if (!ctx_gguf) {
            fprintf(stderr, "\n%s:  failed to load input GGUF from %s\n", __func__, split_params.input.c_str());
            exit(EXIT_FAILURE);
        }
        ctx_ggufs.push_back(ctx_gguf);
        ctx_metas.push_back(ctx_meta);

        if (i_split == 0) {
            auto key_n_split = gguf_find_key(ctx_gguf, LLM_KV_SPLIT_COUNT);
            if (key_n_split < 0) {
                fprintf(stderr,
                        "\n%s: input file does not contain %s metadata\n",
                        __func__,
                        LLM_KV_SPLIT_COUNT);
                gguf_free(ctx_gguf);
                ggml_free(ctx_meta);
                gguf_free(ctx_out);
                fout.close();
                exit(EXIT_FAILURE);
            }

            n_split = gguf_get_val_u16(ctx_gguf, key_n_split);
            if (n_split < 1) {
                fprintf(stderr,
                        "\n%s: input file does not contain a valid split count %d\n",
                        __func__,
                        n_split);
                gguf_free(ctx_gguf);
                ggml_free(ctx_meta);
                gguf_free(ctx_out);
                fout.close();
                exit(EXIT_FAILURE);
            }

            // Verify the file naming and extract split_prefix
            if (!llama_split_prefix(split_prefix, sizeof (split_prefix), split_path, i_split, n_split)) {
                fprintf(stderr, "\n%s: unexpected input file name: %s"
                                " i_split=%d"
                                " n_split=%d\n", __func__,
                        split_path, i_split, n_split);
                gguf_free(ctx_gguf);
                ggml_free(ctx_meta);
                gguf_free(ctx_out);
                fout.close();
                exit(EXIT_FAILURE);
            }

            // Do not trigger merge if we try to merge again the output
            gguf_set_val_u16(ctx_gguf, LLM_KV_SPLIT_COUNT, 0);

            // Set metadata from the first split
            gguf_set_kv(ctx_out, ctx_gguf);
        }

        auto n_tensors = gguf_get_n_tensors(ctx_gguf);
        for (int i_tensor = 0; i_tensor < n_tensors; i_tensor++) {
            const char * t_name = gguf_get_tensor_name(ctx_gguf, i_tensor);
            struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);
            gguf_add_tensor(ctx_out, t);
        }
        total_tensors += n_tensors;

        fprintf(stderr, "\033[3Ddone\n");
    }

    // placeholder for the meta data
    {
        auto meta_size = gguf_get_meta_size(ctx_out);
        ::zeros(fout, meta_size);
    }

    // Write tensors data
    for (int i_split = 0; i_split < n_split; i_split++) {
        llama_split_path(split_path, sizeof(split_path), split_prefix, i_split, n_split);
        std::ifstream f_input(split_path, std::ios::binary);
        if (!f_input.is_open()) {
            fprintf(stderr, "%s:  failed to open input GGUF from %s\n", __func__, split_path);
            for (uint32_t i = 0; i < ctx_ggufs.size(); i++) {
                gguf_free(ctx_ggufs[i]);
                ggml_free(ctx_metas[i]);
            }
            gguf_free(ctx_out);
            fout.close();
            exit(EXIT_FAILURE);
        }
        fprintf(stderr, "%s: writing tensors %s ...", __func__, split_path);

        auto * ctx_gguf = ctx_ggufs[i_split];
        auto * ctx_meta = ctx_metas[i_split];

        auto n_tensors = gguf_get_n_tensors(ctx_gguf);
        for (int i_tensor = 0; i_tensor < n_tensors; i_tensor++) {
            const char * t_name = gguf_get_tensor_name(ctx_gguf, i_tensor);
            struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);

            auto n_bytes = ggml_nbytes(t);

            if (read_data.size() < n_bytes) {
                read_data.resize(n_bytes);
            }

            auto offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i_tensor);
            f_input.seekg(offset);
            f_input.read((char *)read_data.data(), n_bytes);

            // write tensor data + padding
            fout.write((const char *)read_data.data(), n_bytes);
            zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
        }

        gguf_free(ctx_gguf);
        ggml_free(ctx_meta);
        f_input.close();
        fprintf(stderr, "\033[3Ddone\n");
    }

    {
        // go back to beginning of file and write the updated metadata
        fout.seekp(0);
        std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
        gguf_get_meta_data(ctx_out, data.data());
        fout.write((const char *)data.data(), data.size());

        fout.close();
        gguf_free(ctx_out);
    }

    fprintf(stderr, "%s: %s merged from %d split with %d tensors.\n",
            __func__, split_params.output.c_str(), n_split, total_tensors);
}

int main(int argc, const char ** argv) {
    if (argc < 3) {
        split_print_usage(argv[0]);
    }

    split_params params;
    params.executable = argv[0];
    split_params_parse(argc, argv, params);

    switch (params.operation) {
        case SPLIT_OP_SPLIT: gguf_split(params);
            break;
        case SPLIT_OP_MERGE: gguf_merge(params);
            break;
        default: split_print_usage(argv[0]);
            exit(EXIT_FAILURE);
    }

    return 0;
}
