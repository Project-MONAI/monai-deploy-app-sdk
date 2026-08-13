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

if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "[FAIL] Virtual environment not found at ${VENV}"
    exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

PYTHON="${VENV}/bin/python"

# Packages to check (raw pip package name)
PACKAGES=(
    "holoscan-cu13"
    "cupy-cuda13x"
    "rmm-cu13"
    "monai"
    "torch"
    "pydicom"
    "highdicom"
)

FAILED=0
PASS=0

echo "=============================================="
echo "  Virtual Environment Package Validation"
echo "  venv: ${VENV}"
echo "=============================================="
echo ""

for pkg in "${PACKAGES[@]}"; do
    version=$("${PYTHON}" -c "
import importlib.metadata, sys
try:
    dist = importlib.metadata.distribution(sys.argv[1])
    print(dist.version)
except importlib.metadata.PackageNotFoundError:
    sys.exit(1)
" "$pkg" 2>/dev/null) || version=""

    if [[ -n "$version" ]]; then
        printf "[OK]   %-20s v%s\n" "$pkg" "$version"
        PASS=$((PASS + 1))
    else
        printf "[MISS] %-20s\n" "$pkg"
        FAILED=$((FAILED + 1))
    fi
done

# Check for conflicting packages
echo ""
echo "  Conflict checks:"
for conflict in "cupy-cuda12x" "holoscan"; do
    if "${PYTHON}" -c "import importlib.metadata; importlib.metadata.distribution('${conflict}')" 2>/dev/null; then
        printf "  [WARN] %-20s is installed (should be removed)\n" "$conflict"
    else
        printf "  [OK]   %-20s not present\n" "$conflict"
    fi
done

echo ""
echo "----------------------------------------------"
echo "  Passed: ${PASS}  Missing: ${FAILED}"
echo "----------------------------------------------"

if [[ ${FAILED} -gt 0 ]]; then
    echo "[FAIL] ${FAILED} required package(s) missing"
    exit 1
fi

echo "[OK]   All required packages are present"
exit 0
