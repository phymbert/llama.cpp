## GGUF split Example

CLI to split / merge GGUF files.

**Command line options:**

- `--split`: split GGUF to multiple GGUF, default operation.
- `--split-max-tensors N`: maximum tensors in each split: default(128)
- `--merge`: merge multiple GGUF to a single GGUF.
- `--split-max-size N(G|M)`: max size of each split: default unused. This is a soft limit. Example 4G or 4096M for four gigabytes.
- `--no-tensor-in-metadata`: the first shard will not contain tensors data but only metadata, default disabled. It can speed up parallel downloads if enabled.
