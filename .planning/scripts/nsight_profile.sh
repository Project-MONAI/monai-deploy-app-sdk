#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Nsight Systems (nsys) profiling harness
#
# Usage:
#   ./nsight_profile.sh <python_script> [script_args...]
#
# Profiles the given Python script with both CPU and GPU timelines.
# Output lands in .planning/profiles/ with a timestamped .nsys-rep filename.
#
# NVTX markers are supported out-of-the-box — just import the companion
# nvtx_markers.py helper from inside the target script:
#
#   from .planning.scripts.nvtx_markers import push_range, pop_range
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Validate arguments ────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <python_script> [script_args...]"
    echo ""
    echo "Profiles a Python script with Nsight Systems (nsys)."
    echo "Output is written to .planning/profiles/<timestamp>.nsys-rep"
    exit 1
fi

TARGET_SCRIPT="$1"
shift

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
    echo "Error: script '${TARGET_SCRIPT}' not found."
    exit 1
fi

# ── Resolve directories (relative to this script's location) ──────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILES_DIR="$(cd "${SCRIPT_DIR}/../profiles" && pwd)"

mkdir -p "${PROFILES_DIR}"

# ── Timestamped output file ──────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BASENAME="$(basename "${TARGET_SCRIPT}" .py)"
OUTPUT_FILE="${PROFILES_DIR}/${BASENAME}_${TIMESTAMP}.nsys-rep"

echo "┌─────────────────────────────────────────────────────────────┐"
echo "│  Nsight Systems Profiler — ${BASENAME}"
echo "├─────────────────────────────────────────────────────────────┤"
echo "│  Script   : ${TARGET_SCRIPT}"
echo "│  Output   : ${OUTPUT_FILE}"
echo "│  CPU/GPU  : both"
echo "│  NVTX     : enabled (import nvtx_markers from target script)"
echo "└─────────────────────────────────────────────────────────────┘"
echo ""

# ── Check for nsys ────────────────────────────────────────────────────────────
if ! command -v nsys &>/dev/null; then
    echo "Error: 'nsys' not found in PATH."
    echo "Install Nsight Systems from: https://developer.nvidia.com/nsight-systems"
    exit 1
fi

# ── Run nsys profile ──────────────────────────────────────────────────────────
# --trace=cuda,nvtx,osrt,cublas,cudnn
#   cuda    — GPU kernel timeline
#   nvtx    — NVTX range markers (annotations)
#   osrt    — OS runtime (CPU threads, syscalls)
#   cublas/cudnn — BLAS/CuDNN API + kernel activity (cuda memcpy via cuda trace)
#
# NOTE: nsys >= 2024 renamed the old 'cub' trace category; 'cub' is no longer
# a valid value (fix 2026-08-17, nsys 2025.6.3).
#
# --capture-range=cudaProfilerApi
#   Start/stop profiling via cudaProfilerStart/Stop — useful when the
#   target script wraps regions with nvtx_markers or torch.cuda.nvtx.
#
# --capture-range-end=stop
#   Stop recording when the last range ends.
nsys profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    -o "${OUTPUT_FILE}" \
    python "${TARGET_SCRIPT}" "$@"

RESULT=$?

# ── Post-run ──────────────────────────────────────────────────────────────────
if [[ ${RESULT} -eq 0 ]]; then
    echo ""
    echo "✓ Profile complete: ${OUTPUT_FILE}"
    echo ""
    echo "Open with:"
    echo "  nsys-ui ${OUTPUT_FILE}"
    echo ""
    echo "Or export to Chrome tracing:"
    echo "  nsys stats ${OUTPUT_FILE}"
    echo "  nsys export --format=chrome-trace ${OUTPUT_FILE}"
else
    echo ""
    echo "✗ Profiling failed (exit code ${RESULT})."
    echo "  Raw output may still exist at: ${OUTPUT_FILE}"
fi

exit ${RESULT}
