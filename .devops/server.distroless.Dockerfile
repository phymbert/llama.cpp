ARG UBUNTU_VERSION=22.04

FROM ubuntu:$UBUNTU_VERSION as build

RUN set -eux; \
    apt-get update; \
    apt-get install -y \
      build-essential \
      pkg-config \
      git \
      cmake \
      libopenblas-dev; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*;

WORKDIR /app

COPY . .

RUN set -eux; \
    mkdir build; \
    cd build; \
    cmake .. \
      -DLLAMA_CUBLAS=ON \
      -DCMAKE_BUILD_TYPE=Release; \
    cmake --build . \
      --config Release \
      --target server;

FROM distroless/cc-debian12:nonroot as runtime

COPY --from=build /usr/lib/x86_64-linux-gnu/libopenblas.so.0 \
                    /usr/lib/x86_64-linux-gnu/libquadmath.so.0 \
                    /usr/lib/x86_64-linux-gnu/libgfortran.so.5 \
                /usr/lib/x86_64-linux-gnu/
COPY --from=build /app/build/bin/server /server

ENV LC_ALL=C.utf8

ENTRYPOINT [ "/server" ]
