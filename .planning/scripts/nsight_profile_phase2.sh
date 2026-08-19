#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 nsys harness — profiles the FAST app's DEFAULT BUNDLE run
# (HOLOSCAN_MODEL_LIST unset = 3d_fullres + 3d_lowres + 3d_cascade_fullres)
# on the airway study (Phase 2 task 2.14 / INFR-005).
#
# Sibling of nsight_profile.sh: same nsys 2025.6.3 flag set, but full-process
# capture (NO --capture-range=cudaProfilerApi — the app never calls
# cudaProfilerStart/Stop, and the setup/warmup span MUST be in the trace for
# the cudaMalloc/RMM churn check).
#
# Usage:
#   ./nsight_profile_phase2.sh
# Output: .planning/profiles/phase2/phase2_bundle_<timestamp>.nsys-rep (+ .sqlite)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_ROOT="${REPO_ROOT}/examples/apps/cchmc-nnunet-fast"
STUDY="${REPO_ROOT}/testdata/airway_input"
MODEL="${REPO_ROOT}/examples/apps/cchmc_nnunet_fifteen_ckpt_app/models"
PY="/tmp/monai-env/.venv/bin/python"
OUT_DIR="${REPO_ROOT}/.planning/profiles/phase2"
NSYS="$(command -v nsys)"

mkdir -p "${OUT_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_FILE="${OUT_DIR}/phase2_bundle_${TIMESTAMP}.nsys-rep"

echo "┌─────────────────────────────────────────────────────────────┐"
echo "│  Phase 2 nsys — fast-app default BUNDLE (airway study)"
echo "├─────────────────────────────────────────────────────────────┤"
echo "│  Output: ${OUTPUT_FILE}"
echo "└─────────────────────────────────────────────────────────────┘"

# Full-process capture; same trace categories as the Phase 0/1 harness
# ('cub' no longer exists in nsys >= 2024 — do not re-add it).
# ulimit -s unlimited: holoscan 32 MB stack requirement (baseline pattern).
bash -c "
    ulimit -s unlimited
    cd '${APP_ROOT}'
    # -u: HOLOSCAN_MODEL_LIST must be UNSET (the app treats an empty value
    # as an explicit empty model list — os.environ.get, not a truthiness check)
    env -u HOLOSCAN_MODEL_LIST PYTHONUNBUFFERED=1 \
        '${NSYS}' profile \
            --trace=cuda,nvtx,osrt,cublas,cudnn \
            -o '${OUTPUT_FILE}' \
            --force-overwrite true \
            '${PY}' -m my_app --input '${STUDY}' --model '${MODEL}' --output /tmp/phase2_nsys_out
"

echo ""
echo "✓ Profile complete: ${OUTPUT_FILE}"
echo "  (sqlite sidecar auto-generated alongside: ${OUTPUT_FILE}.sqlite)"
