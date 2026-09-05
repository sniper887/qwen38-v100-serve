#!/usr/bin/env bash
# ==============================================================================
# check-env.sh — Environment verification for Qwen3.8-27B on NVIDIA Volta (sm_70)
# ==============================================================================
# Validates driver version, CUDA toolkit, host compiler, GPU architecture,
# build tools, and model paths against stack-requirements.txt.
#
# Usage:
#   ./check-env.sh
# ==============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration if available
for env_file in "${SCRIPT_DIR}/.env" "${SCRIPT_DIR}/serve.env"; do
    if [[ -f "${env_file}" ]]; then
        # shellcheck disable=SC1090
        source "${env_file}"
        break
    fi
done

PASS="[  OK  ]"
WARN="[ WARN ]"
FAIL="[ FAIL ]"
INFO="[ INFO ]"

ERRORS=0
WARNINGS=0

echo "=============================================================================="
echo " Environment Doctor: Qwen3.8-27B Serving Stack"
echo " Target Hardware: NVIDIA Volta sm_70 (e.g. Tesla V100 32GB)"
echo "=============================================================================="
echo ""

# ------------------------------------------------------------------------------
# 1. NVIDIA Driver Check
# ------------------------------------------------------------------------------
echo "--- 1. NVIDIA Driver & GPU Hardware ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")"
    DRIVER_MAJOR="$(echo "${DRIVER_VER}" | cut -d. -f1)"
    echo "${INFO} Detected NVIDIA Driver version: ${DRIVER_VER}"

    if [[ "${DRIVER_MAJOR}" =~ ^[0-9]+$ ]]; then
        if (( DRIVER_MAJOR > 580 )); then
            echo "${FAIL} NVIDIA driver branch ${DRIVER_MAJOR}xx drops Volta (sm_70) support!"
            echo "       Volta requires driver branch <= 580xx (e.g. 580.142 or 535.xx)."
            ERRORS=$((ERRORS + 1))
        elif (( DRIVER_MAJOR < 525 )); then
            echo "${WARN} NVIDIA driver version ${DRIVER_VER} is older than recommended (>= 525.60)."
            WARNINGS=$((WARNINGS + 1))
        else
            echo "${PASS} NVIDIA driver version ${DRIVER_VER} is compatible with Volta sm_70."
        fi
    fi

    # Check GPU device(s)
    GPU_COUNT="$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -n1 || echo 1)"
    TARGET_DEV="${CUDA_VISIBLE_DEVICES:-0}"
    # take first device in comma-separated list
    TARGET_INDEX="${TARGET_DEV%%,*}"

    GPU_NAME="$(nvidia-smi -i "${TARGET_INDEX}" --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")"
    GPU_MEM="$(nvidia-smi -i "${TARGET_INDEX}" --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "unknown")"
    echo "${INFO} Target GPU ${TARGET_INDEX}: ${GPU_NAME} (${GPU_MEM})"

    if [[ "${GPU_NAME}" =~ "V100" || "${GPU_NAME}" =~ "Titan V" ]]; then
        echo "${PASS} Target GPU ${TARGET_INDEX} is a Volta architecture GPU (sm_70)."
    else
        echo "${WARN} Target GPU ${TARGET_INDEX} (${GPU_NAME}) is not verified as Volta sm_70."
        echo "       Ensure CUDA_VISIBLE_DEVICES points to your sm_70 device."
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "${FAIL} 'nvidia-smi' not found on PATH. NVIDIA GPU driver may not be installed."
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ------------------------------------------------------------------------------
# 2. CUDA Toolkit & Host Compiler
# ------------------------------------------------------------------------------
echo "--- 2. CUDA Toolkit & Host Compiler ---"

NVCC_BIN=""
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    NVCC_BIN="${CUDA_HOME}/bin/nvcc"
elif command -v nvcc >/dev/null 2>&1; then
    NVCC_BIN="$(command -v nvcc)"
elif [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
    NVCC_BIN="/usr/local/cuda/bin/nvcc"
fi

if [[ -n "${NVCC_BIN}" ]]; then
    CUDA_VER_RAW="$("${NVCC_BIN}" --version | grep "release" | sed -E 's/.*release ([0-9]+\.[0-9]+).*/\1/' || echo "unknown")"
    echo "${INFO} Found nvcc at ${NVCC_BIN} (Release ${CUDA_VER_RAW})"
    CUDA_MAJOR="$(echo "${CUDA_VER_RAW}" | cut -d. -f1)"

    if [[ "${CUDA_MAJOR}" =~ ^[0-9]+$ ]]; then
        if (( CUDA_MAJOR >= 13 )); then
            echo "${FAIL} CUDA ${CUDA_VER_RAW} detected! CUDA 13.x dropped sm_70 compilation entirely."
            echo "       You must install CUDA 12.x (e.g. 12.0 - 12.9) and set CUDA_HOME."
            ERRORS=$((ERRORS + 1))
        elif (( CUDA_MAJOR < 12 )); then
            echo "${WARN} CUDA ${CUDA_VER_RAW} is older than tested CUDA 12.x."
            WARNINGS=$((WARNINGS + 1))
        else
            echo "${PASS} CUDA Toolkit version ${CUDA_VER_RAW} supports sm_70 compilation."
        fi
    fi
else
    echo "${WARN} 'nvcc' not found. You can still run pre-built binaries, but compilation requires CUDA 12.x."
    WARNINGS=$((WARNINGS + 1))
fi

GCC_BIN=""
if [[ -n "${GCC_HOME:-}" && -x "${GCC_HOME}/bin/gcc" ]]; then
    GCC_BIN="${GCC_HOME}/bin/gcc"
elif [[ -n "${CC:-}" && -x "${CC}" ]]; then
    GCC_BIN="${CC}"
elif command -v gcc >/dev/null 2>&1; then
    GCC_BIN="$(command -v gcc)"
fi

if [[ -n "${GCC_BIN}" ]]; then
    GCC_VER_RAW="$("${GCC_BIN}" -dumpfullversion -dumpversion 2>/dev/null || "${GCC_BIN}" -dumpversion || echo "unknown")"
    echo "${INFO} Found C compiler at ${GCC_BIN} (Version ${GCC_VER_RAW})"
    GCC_MAJOR="$(echo "${GCC_VER_RAW}" | cut -d. -f1)"

    if [[ "${GCC_MAJOR}" =~ ^[0-9]+$ ]]; then
        if (( GCC_MAJOR >= 14 )); then
            echo "${WARN} GCC ${GCC_VER_RAW} may be incompatible with CUDA 12 nvcc."
            echo "       If building from source fails, use GCC 11, 12, or 13 via GCC_HOME or CC/CXX."
            WARNINGS=$((WARNINGS + 1))
        else
            echo "${PASS} GCC version ${GCC_VER_RAW} is compatible with CUDA 12.x."
        fi
    fi
else
    echo "${WARN} C compiler (gcc) not found on PATH."
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# ------------------------------------------------------------------------------
# 3. Build Utilities
# ------------------------------------------------------------------------------
echo "--- 3. Build Utilities ---"
for tool in cmake git; do
    if command -v "${tool}" >/dev/null 2>&1; then
        tool_ver="$("${tool}" --version | head -n1)"
        echo "${PASS} ${tool}: ${tool_ver}"
    else
        echo "${WARN} '${tool}' not found. Needed if compiling llama.cpp from source."
        WARNINGS=$((WARNINGS + 1))
    fi
done

if command -v ccache >/dev/null 2>&1; then
    echo "${PASS} ccache: available (accelerates rebuilds)"
else
    echo "${INFO} ccache: not found (optional, install for faster rebuilds)"
fi
echo ""

# ------------------------------------------------------------------------------
# 4. Binaries & Serving Artifacts
# ------------------------------------------------------------------------------
echo "--- 4. Pre-built Binaries ---"
BUILDS_DIR="${BUILDS_DIR:-${SCRIPT_DIR}/builds}"

STOCK_BIN="${BUILDS_DIR}/stock/bin/llama-server"
[[ ! -x "${STOCK_BIN}" ]] && STOCK_BIN="${BUILDS_DIR}/default/bin/llama-server"

PATCHED_BIN="${BUILDS_DIR}/patched/bin/llama-server"
[[ ! -x "${PATCHED_BIN}" ]] && PATCHED_BIN="${BUILDS_DIR}/T2-001-gqa-packing/bin/llama-server"

if [[ -x "${STOCK_BIN}" ]]; then
    echo "${PASS} Stock llama-server found at: ${STOCK_BIN}"
else
    echo "${WARN} Stock llama-server not found (run './build.sh --stock' or './build.sh --all')"
    WARNINGS=$((WARNINGS + 1))
fi

if [[ -x "${PATCHED_BIN}" ]]; then
    echo "${PASS} Patched (T2-001) llama-server found at: ${PATCHED_BIN}"
else
    echo "${WARN} Patched llama-server not found (run './build.sh --patched' or './build.sh --all')"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# ------------------------------------------------------------------------------
# 5. Model Verification
# ------------------------------------------------------------------------------
echo "--- 5. Model File ---"
MODEL_PATH="${MODEL:-}"
if [[ -z "${MODEL_PATH}" ]]; then
    # Look in common fallback paths
    for cand in \
        "/ssd-storage/llm-models/Qwen3.8-27B-UD-Q4_K_M.gguf" \
        "/storage/llm-models/Qwen3.8-27B-UD-Q4_K_M.gguf" \
        "${SCRIPT_DIR}/models/Qwen3.8-27B-UD-Q4_K_M.gguf"; do
        if [[ -f "${cand}" ]]; then
            MODEL_PATH="${cand}"
            break
        fi
    done
fi

if [[ -n "${MODEL_PATH}" && -f "${MODEL_PATH}" ]]; then
    MODEL_SIZE_GB="$(du -h "${MODEL_PATH}" | cut -f1)"
    echo "${PASS} Model found: ${MODEL_PATH} (${MODEL_SIZE_GB})"
else
    echo "${WARN} Model file not specified or not found."
    echo "       Set MODEL in serve.env or pass '--model /path/to/model.gguf' to serve.sh."
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo "=============================================================================="
echo " Summary: ${ERRORS} error(s), ${WARNINGS} warning(s)"
echo "=============================================================================="

if (( ERRORS > 0 )); then
    echo "Result: Environment does not meet critical stack requirements."
    echo "        Resolve the errors above before running."
    exit 1
elif (( WARNINGS > 0 )); then
    echo "Result: Environment is functional, but review warnings above."
    exit 0
else
    echo "Result: Environment fully matches stack requirements! Ready to build and serve."
    exit 0
fi
