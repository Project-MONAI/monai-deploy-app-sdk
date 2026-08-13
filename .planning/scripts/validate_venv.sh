#!/usr/bin/env bash
#
# Validate that the monai-env virtual environment has all required cu13 packages.
#
# Usage:
#   bash .planning/scripts/validate_venv.sh
#
# Exits 0 if every required package is present, 1 otherwise.
#
set -euo pipefail

VENV="/tmp/monai-env/.venv"

# Activate the virtual environment (load PIP_PREFIX_DIR for `pip list`)
# shellcheck disable=SC1091
if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "[FAIL] Virtual environment not found at ${VENV}"
    exit 1
fi
source "${VENV}/bin/activate"

PYTHON="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

# Packages we expect — each value is a display-friendly hint
declare -A EXPECTED=(
    ["holoscan-cu13"]="holoscan-cu13"
    ["cupy-cuda13x"]="cupy-cuda13x"
    ["rmm-cu13"]="rmm-cu13"
    ["monai"]="MONAI"
    ["torch"]="PyTorch"
    ["pydicom"]="pydicom"
    ["highdicom"]="highdicom"
)

FAILED=0
PASS=0

echo "=============================================="
echo "  Virtual Environment Package Validation"
echo "  venv: ${VENV}"
echo "=============================================="
echo ""

for pkg in "${!EXPECTED[@]}"; do
    label="${EXPECTED[$pkg]}"
    # pip list --format=json outputs one JSON object per line
    result=$("${PIP}" list --format=json 2>/dev/null | python3 -c "
import sys, json
target = sys.argv[1]
for line in sys.stdin:
    obj = json.loads(line)
    name = obj['name'].lower().replace('-', '').replace('_', '').replace('.', '')
    if name == target.lower().replace('-', '').replace('_', '').replace('.', ''):
        print(obj['version'])
        sys.exit(0)
sys.exit(1)
" "$pkg" 2>/dev/null) || result=""

    if [[ -n "$result" ]]; then
        printf "[OK]   %-20s v%s\n" "$label" "$result"
        ((PASS++))
    else
        printf "[MISS] %-20s\n" "$label"
        ((FAILED++))
    fi
done

echo ""
echo "----------------------------------------------"
echo "  Passed: ${PASS}  Missing: ${FAILED}"
echo "----------------------------------------------"

if [[ ${FAILED} -gt 0 ]]; then
    echo "[FAIL] ${FAILED} required package(s) missing or not detectable"
    exit 1
fi

echo "[OK]   All required packages are present"
exit 0
