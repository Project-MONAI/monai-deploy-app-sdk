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
# Security model: the archive is downloaded to a private 0600 temp file.
# Entry names are validated BEFORE extraction (absolute paths, any ".."
# component, and symlink/special entries are rejected). Extraction goes to
# a private 0700 temp dir. Checksums are verified BEFORE the existing
# models/ directory is touched — fail closed against the git checkout's
# MANIFEST.sha256: an archive-internal manifest is self-referential
# (whoever controls the zip controls the manifest), so it is never trusted
# and the script exits 1 when the checkout manifest is absent. Only files
# listed in the manifest are installed (unlisted payloads are rejected).
# models/ is then installed swap-with-rollback: a unique staging dir beside
# models/ (same filesystem, so the final rename is atomic), the existing
# installation kept in place until the new one lands, restored
# automatically on failure.
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

# Private per-run staging: mktemp creates with O_EXCL + 0600/0700, so a
# pre-planted path cannot redirect a write and concurrent runs cannot clobber
# each other. The EXIT trap cleans up on any exit path.
ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/tcv-model-bundle.XXXXXX.zip")"
VERIFY_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tcv-model-verify.XXXXXX")"
trap 'rm -f "$ARCHIVE"; rm -rf "$VERIFY_DIR"' EXIT

echo "Downloading ${URL} ..."
# Zenodo serves public record files directly (HTTP 200, no redirect).
# --max-redirs 0 keeps the transfer on the original HTTPS origin.
if command -v curl >/dev/null 2>&1; then
    curl --fail --proto '=https' --proto-redir '=https' --max-redirs 0 -o "$ARCHIVE" "$URL"
else
    wget --max-redirect=0 -O "$ARCHIVE" "$URL"
fi

# Validate entry names BEFORE extraction: absolute paths, any ".." path
# component (including a trailing "/.."), and symlink/special entries are
# rejected — a post-extraction check cannot undo an already-written file.
while IFS= read -r entry; do
    case "/${entry#/}" in
        */.. | */../*) echo "ERROR: unsafe archive path: $entry" >&2; exit 1 ;;
    esac
    case "$entry" in
        /*) echo "ERROR: unsafe archive path: $entry" >&2; exit 1 ;;
    esac
done < <(unzip -Z1 "$ARCHIVE")
if unzip -Z "$ARCHIVE" | awk 'NR > 2 {print $1}' | grep '^[lbcp]' >/dev/null; then
    echo "ERROR: archive contains symlink/special entries; refusing" >&2
    exit 1
fi

echo "Extracting into private verify dir ${VERIFY_DIR} ..."
unzip -q "$ARCHIVE" -d "$VERIFY_DIR"
if find "$VERIFY_DIR" -type l -print -quit | grep -q .; then
    echo "ERROR: archive contains symlink entries; refusing" >&2
    exit 1
fi

# Locate the models root (the directory that contains 3d_fullres/).
CANDIDATE="$(find "$VERIFY_DIR" -maxdepth 3 -type d -name '3d_fullres' -print -quit)"
if [[ -z "$CANDIDATE" ]]; then
    echo "ERROR: expected a models/3d_fullres directory in the archive; not found." >&2
    exit 1
fi
MODELS_SRC="$(dirname "$CANDIDATE")" # the models/ dir itself
ARCH_ROOT="$(dirname "$MODELS_SRC")" # dir containing models/ (manifest CWD)

# Verify checksums BEFORE touching the existing models/. The manifest MUST
# come from the git checkout: a manifest carried inside the archive is
# self-referential (whoever controls the zip controls the manifest), so
# verifying against it would prove nothing — fail closed instead.
if [[ ! -f "${APP_DIR}/MANIFEST.sha256" ]]; then
    echo "ERROR: ${APP_DIR}/MANIFEST.sha256 not found in the checkout; refusing to trust an archive-internal manifest." >&2
    exit 1
fi
echo "Verifying checksums against the checkout manifest (MANIFEST.sha256) ..."
(cd "$ARCH_ROOT" && sha256sum --check "${APP_DIR}/MANIFEST.sha256")

# Fail closed on any file the checkout manifest does not cover: an unlisted
# payload (e.g. an extra model dir the runtime would pick up) must not join
# the install.
UNLISTED="$(cd "$ARCH_ROOT" && comm -23 \
    <(find models -type f | sort) \
    <(awk '{print $2}' "${APP_DIR}/MANIFEST.sha256" | sort -u))"
if [[ -n "$UNLISTED" ]]; then
    echo "ERROR: archive contains files not covered by MANIFEST.sha256:" >&2
    printf '%s\n' "$UNLISTED" >&2
    exit 1
fi

# Every check passed: install with rollback. Stage in a UNIQUE directory
# beside models/ (same filesystem, so the final mv is an atomic rename),
# keep the existing installation available until the new one is in place,
# and restore the old directory if the install fails.
STAGE_NEW="$(mktemp -d "${MODELS_DIR}.staging.XXXXXX")"
trap 'rm -f "$ARCHIVE"; rm -rf "$VERIFY_DIR" "$STAGE_NEW"' EXIT
cp -a "$MODELS_SRC" "${STAGE_NEW}/models"

OLD_DIR=""
if [[ -e "$MODELS_DIR" ]]; then
    # Unique backup name per run: a stale models.old.* left by an interrupted
    # run must never swallow the current models/ (a fixed-name mv would nest it).
    OLD_DIR="$(mktemp -u "${MODELS_DIR}.old.XXXXXX")"
    mv "$MODELS_DIR" "$OLD_DIR"
fi
if mv "${STAGE_NEW}/models" "$MODELS_DIR"; then
    if [[ -n "$OLD_DIR" ]]; then
        rm -rf "$OLD_DIR"
    fi
else
    # Rollback: the new install did not land — restore the previous one.
    if [[ -n "$OLD_DIR" && -e "$OLD_DIR" ]]; then
        mv "$OLD_DIR" "$MODELS_DIR"
    fi
    echo "ERROR: failed to install ${MODELS_DIR}; previous installation restored." >&2
    exit 1
fi

echo "OK: model bundle in place at ${MODELS_DIR}"
