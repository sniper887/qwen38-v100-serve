#!/usr/bin/env bash
# ==============================================================================
# lock-clocks.sh — GPU Clock Management Utility for Reproducible Throughput
# ==============================================================================
# Locks graphics and memory clocks to their maximum supported frequencies
# to prevent dynamic clock throttling, thermal drift, and power fluctuations.
#
# Usage:
#   sudo ./lock-clocks.sh lock            # Lock clocks on GPU 0 (run once per boot)
#   sudo ./lock-clocks.sh -i 1 lock       # Lock clocks on GPU 1
#   sudo ./lock-clocks.sh unlock          # Reset clocks to default behavior
#   ./lock-clocks.sh status               # Check whether clocks are currently locked
#
# Subcommands:
#   lock      Enable persistence mode and lock graphics + memory clocks to max
#   unlock    Disable persistence mode and reset locked clocks to defaults
#   status    Report whether persistence mode and clock locks are active
#             (exits 0 if locked, 1 if unlocked)
# ==============================================================================
set -euo pipefail

GPU_INDEX="${GPU_INDEX:-0}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-i <gpu_index>] <lock|unlock|status> [-h|--help]

Options:
  -i, --gpu <index>   Target GPU index (default: 0)
  -h, --help          Show this message

Subcommands:
  lock                Lock graphics and memory clocks to maximum values.
                      Requires sudo.
  unlock              Reset clocks to default dynamic scaling.
                      Requires sudo.
  status              Check whether clocks are currently locked.
                      Exits 0 if locked, non-zero otherwise.
EOF
}

# Parse optional leading flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--gpu|--index)
            GPU_INDEX="$2"
            shift 2
            ;;
        lock|unlock|status|-h|--help)
            break
            ;;
        *)
            echo "ERROR: Unknown option '$1'" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

ACTION="$1"
shift

require_nvidia_smi() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi not found on PATH." >&2
        exit 1
    fi
}

require_sudo() {
    if ! command -v sudo >/dev/null 2>&1; then
        echo "ERROR: sudo not found. Root privileges required to set clock locks." >&2
        exit 1
    fi
    if ! sudo -n true 2>/dev/null; then
        echo "Elevated permissions required. You may be prompted for your password." >&2
    fi
}

get_max_clock() {
    # $1: gr | mem
    local field="$1"
    nvidia-smi -i "${GPU_INDEX}" --query-gpu="clocks.max.${field}" --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

cmd_lock() {
    require_nvidia_smi
    require_sudo

    local max_gr max_mem
    max_gr="$(get_max_clock gr)"
    max_mem="$(get_max_clock mem)"

    if [[ -z "${max_gr}" || -z "${max_mem}" ]]; then
        echo "ERROR: Failed to query max clocks for GPU ${GPU_INDEX}." >&2
        exit 1
    fi

    echo "Configuring GPU ${GPU_INDEX} for locked performance..."
    echo "  1. Enabling persistence mode..."
    sudo nvidia-smi -i "${GPU_INDEX}" -pm 1

    echo "  2. Locking graphics clock to max (${max_gr} MHz)..."
    sudo nvidia-smi -i "${GPU_INDEX}" -lgc "${max_gr}"

    echo "  3. Locking memory clock to max (${max_mem} MHz)..."
    sudo nvidia-smi -i "${GPU_INDEX}" -lmc "${max_mem}"

    echo "Success: GPU ${GPU_INDEX} clocks locked. Verification: $(basename "$0") -i ${GPU_INDEX} status"
}

cmd_unlock() {
    require_nvidia_smi
    require_sudo

    echo "Resetting GPU ${GPU_INDEX} to dynamic clock management..."
    sudo nvidia-smi -i "${GPU_INDEX}" -rgc || true
    sudo nvidia-smi -i "${GPU_INDEX}" -rmc || true
    sudo nvidia-smi -i "${GPU_INDEX}" -pm 0 || true
    echo "Success: GPU ${GPU_INDEX} clocks reset to dynamic."
}

cmd_status() {
    require_nvidia_smi

    local pm cur_gr cur_mem max_gr max_mem
    pm="$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null | tr -d ' ' || echo "Unknown")"
    cur_gr="$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=clocks.gr --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo "0")"
    cur_mem="$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=clocks.mem --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo "0")"
    max_gr="$(get_max_clock gr)"
    max_mem="$(get_max_clock mem)"

    echo "GPU ${GPU_INDEX} Status:"
    echo "  Persistence Mode: ${pm}"
    echo "  Graphics Clock:   ${cur_gr} MHz (max: ${max_gr} MHz)"
    echo "  Memory Clock:     ${cur_mem} MHz (max: ${max_mem} MHz)"

    if [[ "${pm}" == "Enabled" && "${cur_gr}" == "${max_gr}" && "${cur_mem}" == "${max_mem}" ]]; then
        echo "STATUS: LOCKED (Maximum throughput sustained)"
        exit 0
    else
        echo "STATUS: NOT LOCKED (Clock throttling or drift may occur)"
        echo "Run: sudo $(basename "$0") -i ${GPU_INDEX} lock"
        exit 1
    fi
}

case "${ACTION}" in
    lock)
        cmd_lock
        ;;
    unlock)
        cmd_unlock
        ;;
    status)
        cmd_status
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "ERROR: Unknown command '${ACTION}'" >&2
        usage
        exit 1
        ;;
esac
