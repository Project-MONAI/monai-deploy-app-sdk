#!/usr/bin/env bash
# Download the cchmc-tcv-fast model bundle from Zenodo into models/.
#
# Usage:
#   bash scripts/download_models.sh
#
# The default Zenodo record (23391210, DOI 10.5281/zenodo.23391210) is baked
# in below. Override with ZENODO_RECORD=<record-number> if the bundle is ever
# republished to a new record (integer ID: zenodo.org/records/<record-number>).
#
# The archive is expected to contain (or extract to):
#   models/3d_fullres/fold_{0..4}/final_model.pt
#   models/3d_fullres/nnunet_checkpoint.pth
#   models/jsonpkls/{plans.json,postprocessing.pkl,dataset.json}
#
# After download, checksums are verified against MANIFEST.sha256.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${APP_DIR}/models"

REC="${ZENODO_RECORD:-23391210}"
if ! [[ "$REC" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ZENODO_RECORD must be a positive integer (got: ${REC})" >&2
    exit 1
fi

URL="https://zenodo.org/records/${REC}/files/model-bundle.zip"
echo "Downloading ${URL} ..."

# Zenodo serves public record files directly (HTTP 200, no redirect);
# redirects are deliberately disabled to keep the transfer HTTPS-only.
if command -v curl >/dev/null 2>&1; then
    curl --fail --proto '=https' --proto-redir '=https' --max-redirs 0 -o /tmp/tcv-model-bundle.zip "$URL"
else
    wget --max-redirect=0 -O /tmp/tcv-model-bundle.zip "$URL"
fi

echo "Extracting into ${APP_DIR} ..."
mkdir -p "$MODELS_DIR"
rm -rf "$MODELS_DIR"
unzip -q /tmp/tcv-model-bundle.zip -d "$APP_DIR"
rm -f /tmp/tcv-model-bundle.zip

# If the archive extracted to a different root, relocate to models/
if [[ ! -d "${MODELS_DIR}/3d_fullres" ]]; then
    CANDIDATE="$(find "$APP_DIR" -maxdepth 3 -type d -name '3d_fullres' | head -1 || true)"
    if [[ -n "$CANDIDATE" ]]; then
        mv "$(dirname "$CANDIDATE")" "$MODELS_DIR"
    else
        echo "ERROR: expected models/3d_fullres after extraction; not found." >&2
        exit 1
    fi
fi

echo "Verifying checksums against MANIFEST.sha256 ..."
(cd "$APP_DIR" && sha256sum -c MANIFEST.sha256)

echo "OK: model bundle in place at ${MODELS_DIR}"
