#!/usr/bin/env bash
# ==============================================================================
# build.sh — Generic build script for optimized llama.cpp on NVIDIA Volta (sm_70)
# ==============================================================================
# Builds two configurations:
#   1. Stock llama.cpp  (pinned at tag b10793) -> optimal for MTP speculative decode
#   2. Patched llama.cpp (T2-001 GQA packing)   -> optimal for non-speculative decode
#
# Usage:
#   ./build.sh              # builds both stock and patched (recommended)
#   ./build.sh --all        # builds both stock and patched
#   ./build.sh --stock      # builds only the stock binary
#   ./build.sh --patched    # builds only the T2-001 patched binary
#   ./build.sh --clean      # cleans build directories
#
# Environment Overrides (can also be specified in .env or serve.env):
#   LLAMACPP_SRC     Path to llama.cpp source checkout (cloned automatically if missing)
#   BUILDS_DIR       Destination for compiled binaries (default: ./builds)
#   CUDA_HOME        Path to CUDA 12.x toolkit (default: system /usr/local/cuda or nvcc in PATH)
#   GCC_HOME         Path to GCC 11-13 toolchain (default: system gcc/g++)
#   CUDA_ARCH        Target CUDA architecture (default: 70)
#   UPSTREAM_URL     Git URL for llama.cpp (default: https://github.com/ggml-org/llama.cpp.git)
#   UPSTREAM_REF     Pinned git tag or commit (default: b10793)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env or serve.env if present
for env_file in "${SCRIPT_DIR}/.env" "${SCRIPT_DIR}/serve.env"; do
    if [[ -f "${env_file}" ]]; then
        # shellcheck disable=SC1090
        source "${env_file}"
        break
    fi
done

# Defaults
BUILDS_DIR="${BUILDS_DIR:-${SCRIPT_DIR}/builds}"
LLAMACPP_SRC="${LLAMACPP_SRC:-${SCRIPT_DIR}/llama.cpp}"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/ggml-org/llama.cpp.git}"
UPSTREAM_REF="${UPSTREAM_REF:-b10793}"
CUDA_ARCH="${CUDA_ARCH:-70}"
PATCH_FILE="${SCRIPT_DIR}/patches/0001-t2-001-gqa-packing-sm70.patch"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--all | --stock | --patched | --clean | -h|--help]

Options:
  --all        Build both stock and patched binaries (default)
  --stock      Build stock llama.cpp only (for MTP speculative decoding)
  --patched    Build patched llama.cpp only (T2-001 GQA packing for non-speculative decode)
  --clean      Remove builds/ and scratch worktrees
  -h, --help   Show this message
EOF
}

# ------------------------------------------------------------------------------
# Toolchain & Environment Resolution
# ------------------------------------------------------------------------------
resolve_toolchain() {
    # 1. Resolve NVCC & CUDA_HOME
    NVCC_BIN=""
    if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        NVCC_BIN="${CUDA_HOME}/bin/nvcc"
    elif command -v nvcc >/dev/null 2>&1; then
        NVCC_BIN="$(command -v nvcc)"
        CUDA_HOME="$(cd "$(dirname "${NVCC_BIN}")/.." && pwd)"
    elif [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
        NVCC_BIN="/usr/local/cuda/bin/nvcc"
        CUDA_HOME="/usr/local/cuda"
    else
        echo "ERROR: CUDA nvcc not found. Please install CUDA 12.x and set CUDA_HOME." >&2
        exit 1
    fi

    # Verify CUDA version (CUDA 13+ dropped sm_70)
    local cuda_ver
    cuda_ver="$("${NVCC_BIN}" --version | grep "release" | sed -E 's/.*release ([0-9]+\.[0-9]+).*/\1/' || true)"
    local cuda_major
    cuda_major="$(echo "${cuda_ver}" | cut -d. -f1)"
    if [[ "${cuda_major}" -ge 13 ]]; then
        echo "ERROR: Detected CUDA ${cuda_ver} at ${NVCC_BIN}." >&2
        echo "       CUDA 13.x dropped sm_70 (Volta) support entirely." >&2
        echo "       You must supply a CUDA 12.x toolkit via CUDA_HOME." >&2
        exit 1
    fi
    echo "[build] Using CUDA toolkit: ${CUDA_HOME} (version ${cuda_ver})"

    # 2. Resolve Host Compiler
    CC_BIN=""
    CXX_BIN=""
    if [[ -n "${GCC_HOME:-}" && -x "${GCC_HOME}/bin/gcc" ]]; then
        CC_BIN="${GCC_HOME}/bin/gcc"
        CXX_BIN="${GCC_HOME}/bin/g++"
    elif [[ -n "${CC:-}" && -n "${CXX:-}" && -x "${CC}" && -x "${CXX}" ]]; then
        CC_BIN="${CC}"
        CXX_BIN="${CXX}"
    elif command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
        CC_BIN="$(command -v gcc)"
        CXX_BIN="$(command -v g++)"
    else
        echo "ERROR: GCC/G++ host compiler not found." >&2
        exit 1
    fi

    local gcc_ver
    gcc_ver="$("${CC_BIN}" -dumpversion || echo "unknown")"
    local gcc_major
    gcc_major="$(echo "${gcc_ver}" | cut -d. -f1)"
    if [[ "${gcc_major}" =~ ^[0-9]+$ ]] && [[ "${gcc_major}" -ge 14 ]]; then
        echo "WARNING: Host GCC is version ${gcc_ver}. GCC 14+ is known to fail with CUDA 12 nvcc." >&2
        echo "         If compilation fails, point GCC_HOME to GCC 11-13." >&2
    fi
    echo "[build] Using host compiler: ${CC_BIN} (version ${gcc_ver})"

    # Export library paths for CUDA runtime
    local cuda_lib="${CUDA_HOME}/targets/x86_64-linux/lib"
    [[ ! -d "${cuda_lib}" ]] && cuda_lib="${CUDA_HOME}/lib64"
    if [[ -d "${cuda_lib}" ]]; then
        export LD_LIBRARY_PATH="${cuda_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi

    if [[ -n "${GCC_HOME:-}" && -d "${GCC_HOME}/lib" ]]; then
        export LD_LIBRARY_PATH="${GCC_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
}

# ------------------------------------------------------------------------------
# Source Checkout
# ------------------------------------------------------------------------------
ensure_source() {
    if [[ -d "${LLAMACPP_SRC}" && -f "${LLAMACPP_SRC}/CMakeLists.txt" ]]; then
        echo "[build] Found llama.cpp source at ${LLAMACPP_SRC}"
        return
    fi

    echo "[build] Cloning upstream llama.cpp (${UPSTREAM_REF}) into ${LLAMACPP_SRC}..."
    mkdir -p "$(dirname "${LLAMACPP_SRC}")"
    git clone --depth 1 --branch "${UPSTREAM_REF}" "${UPSTREAM_URL}" "${LLAMACPP_SRC}" || {
        # Fallback if UPSTREAM_REF is a commit SHA rather than tag
        git clone "${UPSTREAM_URL}" "${LLAMACPP_SRC}"
        git -C "${LLAMACPP_SRC}" checkout "${UPSTREAM_REF}"
    }
    echo "[build] Successfully cloned llama.cpp at ${LLAMACPP_SRC}"
}

# ------------------------------------------------------------------------------
# Compilation Routine
# ------------------------------------------------------------------------------
compile_variant() {
    local variant="$1"    # stock | patched
    local src_dir="$2"
    local out_dir="${BUILDS_DIR}/${variant}"

    echo "=============================================================================="
    echo "[build] Configuring and building variant: ${variant}"
    echo "[build] Source: ${src_dir}"
    echo "[build] Output: ${out_dir}"
    echo "=============================================================================="

    mkdir -p "${out_dir}"

    local cmake_args=(
        cmake -B "${out_dir}" -S "${src_dir}"
        -DCMAKE_BUILD_TYPE=Release
        -DGGML_CUDA=ON
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
        -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"
        -DCMAKE_CUDA_HOST_COMPILER="${CC_BIN}"
        -DCMAKE_C_COMPILER="${CC_BIN}"
        -DCMAKE_CXX_COMPILER="${CXX_BIN}"
    )

    if command -v ccache >/dev/null 2>&1; then
        echo "[build] ccache detected; enabling compiler launcher"
        cmake_args+=(
            -DCMAKE_C_COMPILER_LAUNCHER=ccache
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
            -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
        )
    fi

    # Run CMake configure
    "${cmake_args[@]}"

    # Run CMake build
    cmake --build "${out_dir}" --config Release -j "$(nproc)" --target llama-server

    echo "[build] Build successful for ${variant}: ${out_dir}/bin/llama-server"
}

# ------------------------------------------------------------------------------
# Build Stock
# ------------------------------------------------------------------------------
build_stock() {
    ensure_source
    compile_variant "stock" "${LLAMACPP_SRC}"
    # Backward-compatible symlink to 'default'
    if [[ ! -e "${BUILDS_DIR}/default" ]]; then
        ln -s "stock" "${BUILDS_DIR}/default" 2>/dev/null || true
    fi
}

# ------------------------------------------------------------------------------
# Build Patched (T2-001)
# ------------------------------------------------------------------------------
build_patched() {
    ensure_source

    if [[ ! -f "${PATCH_FILE}" ]]; then
        echo "ERROR: Patch file not found at ${PATCH_FILE}" >&2
        exit 1
    fi

    local worktree_dir="${BUILDS_DIR}/.worktrees/t2-001"
    mkdir -p "$(dirname "${worktree_dir}")"

    # Create an isolated git worktree if needed
    if [[ ! -d "${worktree_dir}" ]]; then
        echo "[build] Creating isolated worktree for patched build at ${worktree_dir}..."
        git -C "${LLAMACPP_SRC}" worktree add -f "${worktree_dir}" "HEAD"
        echo "[build] Applying patch: ${PATCH_FILE}..."
        git -C "${worktree_dir}" apply "${PATCH_FILE}"
    else
        echo "[build] Reusing existing worktree at ${worktree_dir}"
    fi

    compile_variant "patched" "${worktree_dir}"
    # Backward-compatible symlink to 'T2-001-gqa-packing'
    if [[ ! -e "${BUILDS_DIR}/T2-001-gqa-packing" ]]; then
        ln -s "patched" "${BUILDS_DIR}/T2-001-gqa-packing" 2>/dev/null || true
    fi
}

# ------------------------------------------------------------------------------
# Clean
# ------------------------------------------------------------------------------
clean_builds() {
    echo "[build] Cleaning builds directory: ${BUILDS_DIR}"
    local worktree_dir="${BUILDS_DIR}/.worktrees/t2-001"
    if [[ -d "${worktree_dir}" && -d "${LLAMACPP_SRC}" ]]; then
        git -C "${LLAMACPP_SRC}" worktree remove -f "${worktree_dir}" 2>/dev/null || true
    fi
    rm -rf "${BUILDS_DIR}"
    echo "[build] Clean complete."
}

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
MODE="all"
if [[ $# -gt 0 ]]; then
    case "$1" in
        --all)     MODE="all" ;;
        --stock)   MODE="stock" ;;
        --patched) MODE="patched" ;;
        --clean)   clean_builds; exit 0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: Unknown option '$1'" >&2; usage; exit 1 ;;
    esac
fi

resolve_toolchain

case "${MODE}" in
    stock)
        build_stock
        ;;
    patched)
        build_patched
        ;;
    all)
        build_stock
        build_patched
        ;;
esac

echo ""
echo "=============================================================================="
echo " Build complete!"
echo " Binaries available at:"
echo "   Stock (MTP on):   ${BUILDS_DIR}/stock/bin/llama-server"
echo "   Patched (no-MTP): ${BUILDS_DIR}/patched/bin/llama-server"
echo " You can now start serving with:"
echo "   ./serve.sh"
echo "=============================================================================="
