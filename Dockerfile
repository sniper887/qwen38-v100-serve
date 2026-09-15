# syntax=docker/dockerfile:1
# ==============================================================================
# Dockerfile — qwen38-v100-serve
# Multi-stage build: compile llama.cpp (stock + patched) on CUDA 12.x, then
# produce a slim runtime image with both binaries and the serve wrapper.
# Target: NVIDIA Volta (sm_70) — Tesla V100 32GB/16GB, Titan V
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies: GCC 12 (CUDA 12-compatible), CMake, Git, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc-12 g++-12 \
        cmake \
        git \
        ccache \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Make GCC 12 the default (GCC 14+ is incompatible with CUDA 12 nvcc headers)
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# CUDA stubs: libcuda.so is not available in the container without NVIDIA driver.
# Create symlink and set LIBRARY_PATH so the linker can find the stub library.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so \
 && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1 2>/dev/null || true
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}
ENV CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs

WORKDIR /build

# Copy project files (build script, patches, serve scripts)
COPY build.sh ./
COPY patches/ ./patches/
COPY serve.env.example ./
COPY serve.sh ./
COPY check-env.sh ./
COPY stack-requirements.txt ./

# Make scripts executable
RUN chmod +x build.sh serve.sh check-env.sh

# Build both stock and patched llama-server binaries for sm_70 (V100)
# - Stock build:    optimal for MTP speculative decoding
# - Patched build:  T2-001 GQA packing, optimal for non-speculative decode
RUN BUILDS_DIR=/build/builds \
    LLAMACPP_SRC=/build/llama.cpp \
    CUDA_ARCH=70 \
    ./build.sh --all

# ------------------------------------------------------------------------------
# Stage 2: Runtime
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy serve scripts and env example
COPY serve.sh ./
COPY serve.env.example ./
COPY check-env.sh ./
RUN chmod +x serve.sh check-env.sh

# Copy compiled binaries from builder
COPY --from=builder /build/builds/stock/bin/llama-server   /app/builds/stock/bin/llama-server
COPY --from=builder /build/builds/patched/bin/llama-server /app/builds/patched/bin/llama-server

# Backward-compatible symlink (T2-001-gqa-packing -> patched)
RUN ln -sf patched /app/builds/T2-001-gqa-packing

# Default environment
ENV BUILDS_DIR=/app/builds \
    HOST=0.0.0.0 \
    PORT=8080 \
    CTX=131072

# Expose the OpenAI-compatible API port
EXPOSE 8080

# Entrypoint: serve.sh picks the right binary per --mtp / --no-mtp
# Usage:
#   docker run --gpus all -v /path/to/models:/models -e MODEL=/models/Qwen3.8-27B.gguf -p 8080:8080 ghcr.io/sniper887/qwen38-v100-serve:latest
#   docker run --gpus all ... ghcr.io/sniper887/qwen38-v100-serve:latest --no-mtp
ENTRYPOINT ["/app/serve.sh"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
