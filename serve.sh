#!/usr/bin/env bash
# ==============================================================================
# serve.sh — Optimized llama-server wrapper for Qwen3.8-27B on NVIDIA Volta (sm_70)
# ==============================================================================
# Two modes, toggleable via --mtp / --no-mtp:
#
#   ./serve.sh              # MTP on  (stock build — fastest: 42.2 tok/s @ 128K)
#   ./serve.sh --no-mtp     # MTP off (T2-001 packed-attention build: 23.8 tok/s @ 128K)
#   ./serve.sh --ctx 65536  # custom context length
#   ./serve.sh --port 8080  # custom HTTP port
#
# The two modes use DIFFERENT BUILDS by design. See README.md for rationale.
#
# Flags:
#   --mtp               Enable MTP speculative decoding (default, uses stock build)
#   --no-mtp            Disable MTP (uses T2-001 packed-attention build)
#   --draft-max N       Maximum MTP draft tokens (default: 3; set to 7 for pure 128K)
#   --model PATH        Path to Qwen3.8-27B GGUF file
#   --ctx N             Context size in tokens (default: 131072)
#   --host H            Host IP to bind (default: 127.0.0.1)
#   --port P            HTTP port to bind (default: 8080)
#   --device D          GPU device ID for CUDA_VISIBLE_DEVICES (default: 0)
#   --builds PATH       Base path to builds directory (default: ./builds)
#   --bin PATH          Direct path to llama-server binary override
#   -h, --help          Show help message
#
# Anything else is passed directly to llama-server.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration file if present (.env or serve.env)
for env_file in "${SCRIPT_DIR}/.env" "${SCRIPT_DIR}/serve.env"; do
    if [[ -f "${env_file}" ]]; then
        # shellcheck disable=SC1090
        source "${env_file}"
        break
    fi
done

# ------------------------------------------------------------------------------
# Default Settings
# ------------------------------------------------------------------------------
BUILDS_DIR="${BUILDS_DIR:-${SCRIPT_DIR}/builds}"
CTX="${CTX:-131072}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
MTP="${MTP:-1}"
DRAFT_MAX="${DRAFT_MAX:-3}"
GPU_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-}"
EXTRA=()

# Model fallback resolution
if [[ -z "${MODEL:-}" ]]; then
    for cand in \
        "/ssd-storage/llm-models/Qwen3.8-27B-UD-Q4_K_M.gguf" \
        "/storage/llm-models/Qwen3.8-27B-UD-Q4_K_M.gguf" \
        "${SCRIPT_DIR}/models/Qwen3.8-27B-UD-Q4_K_M.gguf"; do
        if [[ -f "${cand}" ]]; then
            MODEL="${cand}"
            break
        fi
    done
fi
MODEL="${MODEL:-}"

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mtp)         MTP=1; shift ;;
        --no-mtp)      MTP=0; shift ;;
        --draft-max)   DRAFT_MAX="$2"; shift 2 ;;
        --ctx)         CTX="$2"; shift 2 ;;
        --port)        PORT="$2"; shift 2 ;;
        --host)        HOST="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --device|--gpu) GPU_DEVICE="$2"; shift 2 ;;
        --builds)      BUILDS_DIR="$2"; shift 2 ;;
        --bin)         LLAMA_SERVER_BIN="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)             EXTRA+=("$1"); shift ;;
    esac
done

# ------------------------------------------------------------------------------
# Toolchain & Dynamic Library Setup
# ------------------------------------------------------------------------------
# Configure LD_LIBRARY_PATH if custom CUDA_HOME or GCC_HOME paths are provided.
# Volta (sm_70) requires CUDA 12.x; on systems where the default CUDA is 13.x+,
# CUDA_HOME must point to a 12.x installation.
if [[ -n "${CUDA_HOME:-}" ]]; then
    if [[ -d "${CUDA_HOME}/targets/x86_64-linux/lib" ]]; then
        export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    elif [[ -d "${CUDA_HOME}/lib64" ]]; then
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
fi

if [[ -n "${GCC_HOME:-}" && -d "${GCC_HOME}/lib" ]]; then
    export LD_LIBRARY_PATH="${GCC_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# ------------------------------------------------------------------------------
# Binary Resolution
# ------------------------------------------------------------------------------
find_binary() {
    local primary_path="$1"
    local secondary_path="$2"
    local tertiary_path="$3"

    if [[ -n "${LLAMA_SERVER_BIN}" && -x "${LLAMA_SERVER_BIN}" ]]; then
        echo "${LLAMA_SERVER_BIN}"
        return 0
    fi
    if [[ -x "${primary_path}" ]]; then
        echo "${primary_path}"
        return 0
    fi
    if [[ -x "${secondary_path}" ]]; then
        echo "${secondary_path}"
        return 0
    fi
    if [[ -n "${tertiary_path}" && -x "${tertiary_path}" ]]; then
        echo "${tertiary_path}"
        return 0
    fi
    if command -v llama-server >/dev/null 2>&1; then
        command -v llama-server
        return 0
    fi
    return 1
}

SPEC=()
if [[ "${MTP}" == "1" ]]; then
    # MTP mode: Speculative decoding drafts tokens and verifies in a batch forward pass.
    # Runs the stock build (unpatched) because MTP's 4-wide verify already amortizes KV reads.
    BIN="$(find_binary \
        "${BUILDS_DIR}/stock/bin/llama-server" \
        "${BUILDS_DIR}/default/bin/llama-server" \
        "/storage/qwen-opt-lab/builds/default/bin/llama-server" || true)"
    SPEC=(--spec-type draft-mtp --spec-draft-n-max "${DRAFT_MAX}")
    MODE="MTP on  (stock build, speculative decoding)"
else
    # Non-speculative mode: T2-001 packs 3 query heads into flash_attn_ext_vec (ncols2=3),
    # reducing KV DRAM traffic by 3x and improving 128K decode by +44.9%.
    BIN="$(find_binary \
        "${BUILDS_DIR}/patched/bin/llama-server" \
        "${BUILDS_DIR}/T2-001-gqa-packing/bin/llama-server" \
        "/storage/qwen-opt-lab/builds/T2-001-gqa-packing/bin/llama-server" || true)"
    SPEC=()
    MODE="MTP off (T2-001 packed-attention build, deterministic decode)"
fi

if [[ -z "${BIN}" || ! -x "${BIN}" ]]; then
    echo "ERROR: Could not locate a valid llama-server binary for: ${MODE}" >&2
    echo "       Expected at: ${BUILDS_DIR}/stock/bin/llama-server (or patched/)" >&2
    echo "       Run './build.sh' to compile, or specify '--bin /path/to/llama-server'." >&2
    exit 1
fi

# Preflight: Prove the binary and dynamic libraries load before starting
if ! "${BIN}" --version >/dev/null 2>&1; then
    echo "ERROR: Preflight failed. Binary at ${BIN} cannot execute." >&2
    echo "       Output:" >&2
    "${BIN}" --version 2>&1 | head -n5 >&2 || true
    echo "" >&2
    echo "       This usually indicates missing CUDA 12 runtime libraries (libcudart.so.12)." >&2
    echo "       Ensure CUDA_HOME points to a valid CUDA 12 installation." >&2
    exit 1
fi

# Validate model file
if [[ -z "${MODEL}" || ! -f "${MODEL}" ]]; then
    echo "ERROR: Model file not found." >&2
    echo "       Specified path: '${MODEL}'" >&2
    echo "       Supply a valid GGUF file via '--model /path/to/model.gguf' or MODEL env var." >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# GPU Clock Check (Advisory)
# ------------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    TARGET_INDEX="${GPU_DEVICE%%,*}"
    if ! nvidia-smi -i "${TARGET_INDEX}" --query-gpu=clocks.max.graphics,clocks.current.graphics \
         --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{exit !($1==$2)}'; then
        echo "WARNING: GPU ${TARGET_INDEX} clocks are not locked; decode throughput will drift." >&2
        echo "         Fix: sudo ./lock-clocks.sh -i ${TARGET_INDEX} lock" >&2
    fi
fi

# ------------------------------------------------------------------------------
# Server Launch
# ------------------------------------------------------------------------------
echo "=============================================================================="
echo " Qwen3.8-27B Serving Daemon"
echo "=============================================================================="
echo " Mode:            ${MODE}"
echo " Binary:          ${BIN}"
echo " Model:           ${MODEL}"
echo " Context:         ${CTX} tokens"
echo " GPU Device:      CUDA_VISIBLE_DEVICES=${GPU_DEVICE}"
echo " Endpoint:        http://${HOST}:${PORT}"
echo " Prompt Cache:    ON (explicit: --cache-prompt)"
echo " Parallel Slots:  1 (mandatory: prevents multi-slot VRAM duplication)"
echo "=============================================================================="

# KEY PARAMETER RATIONALE:
# - CUDA_VISIBLE_DEVICES="${GPU_DEVICE}": Mandatory to isolate the Volta sm_70 GPU
#   and prevent llama-server from attempting to run on non-sm_70 display adapters.
# - -ngl 999: Offloads all 65 layers (including blk.64 MTP head) to GPU VRAM.
# - -fa on: Enables FlashAttention kernel execution on sm_70.
# - -b 2048 -ub 2048: Optimal batch/ubatch sizing for Volta memory bandwidth.
# - -ctk f16 -ctv f16: F16 KV cache required for numerical stability and T2-001 head packing.
# - --parallel 1: Mandatory. Default (4 slots) allocates n_ctx per slot, causing OOM at 128K.
# - --cache-prompt: Preserves KV state across multi-turn sessions (22.9s -> 3.08s per turn).
exec env CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" "${BIN}" \
    -m "${MODEL}" \
    -ngl 999 \
    -fa on \
    -b 2048 \
    -ub 2048 \
    -ctk f16 \
    -ctv f16 \
    -mg 0 \
    -c "${CTX}" \
    --parallel 1 \
    --cache-prompt \
    "${SPEC[@]}" \
    --host "${HOST}" \
    --port "${PORT}" \
    "${EXTRA[@]}"
