# cchmc-tcv-fast

Research/evaluation use only — not a cleared clinical device.

Segments **heart, left lung, right lung** from chest CT. A Holoscan-native
MONAI Deploy App Package (MAP) running a 5-fold `3d_fullres` nnUNet ensemble
fine-tuned on the TCV cardiac/lung dataset. Two modes: one-shot **batch** and a
persistent **serve** gRPC daemon.

- Labels: `1 heart` · `2 l_lung` · `3 r_lung`
- Output: one **DICOM-SEG** + one **DICOM-SR** (per-structure volumes) per study

> **For coding agents:** every command below is copy-paste runnable. The only
> environment variable that matters is `VENV`. Set it once (see step 0) and
> run the rest verbatim.

---

## 0. Prerequisites

| Need | Value |
|---|---|
| GPU | 1× NVIDIA with ≥ 20 GiB free VRAM (the baked manifest requests `gpuMemory: 20Gi`); pick **any free GPU** and substitute it for `<N>` below |
| Docker | with `buildx` + NVIDIA Container Toolkit |
| `monai-deploy` CLI | this repo (editable install in the build venv). The PyPI SDK predates the CUDA-13 packager support this bake relies on — do **not** substitute `pip install monai-deploy-app-sdk` |
| Python venv (build/clients) | a dedicated venv with the editable SDK install + client deps (see below) |

```bash
# Point VENV at the build venv (it carries the editable SDK install +
# client deps). Example:
python3 -m venv .venv && source .venv/bin/activate
pip install -e <repo-root>
pip install grpcio==1.83.1 protobuf==6.33.6
VENV=$VIRTUAL_ENV
```

- **Do not** use the system Python. **Do not** `pip install nnunetv2` from PyPI
  in the build venv — the vendored editable copy under the repo root (`nnUNet/`)
  is the canonical source; the MAP image pins PyPI `nnunetv2==2.8.1`.

---

## 0b. Download model weights

The 5.3 GB weight bundle is **not** in git. Download it from Zenodo
([record 23391210](https://zenodo.org/records/23391210), DOI
[10.5281/zenodo.23391210](https://doi.org/10.5281/zenodo.23391210)):

```bash
cd examples/apps/cchmc-tcv-fast
bash scripts/download_models.sh
# (override with ZENODO_RECORD=<number> if the bundle moves to a new record)
```

The script downloads and extracts the bundle into `models/`, then verifies
checksums against `MANIFEST.sha256`.

Expected layout after download:

```
models/
├── 3d_fullres/
│   ├── fold_0/final_model.pt
│   ├── fold_1/final_model.pt
│   ├── fold_2/final_model.pt
│   ├── fold_3/final_model.pt
│   ├── fold_4/final_model.pt
│   └── nnunet_checkpoint.pth
└── jsonpkls/
    ├── plans.json
    ├── postprocessing.pkl
    └── dataset.json
```

---

## 1. Build the MAP image (once)

```bash
cd <repo>/examples/apps/cchmc-tcv-fast
WHEEL_DIR=$(mktemp -d)

# 1a. Build the SDK wheel from this repo. (Verified: the public PyPI SDK's
# packager predates CUDA-13 holoscan handling and bakes a broken CPU
# holoscan — use this repo's packager.)
( cd ../../../ && "$VENV/bin/python" -m pip wheel . --no-deps -w "$WHEEL_DIR" )
WHEEL=$(ls "$WHEEL_DIR"/monai_deploy_app_sdk-*.whl)

# 1b. Bake. ~6 min. Produces:
# cchmc-tcv-fast-x64-workstation-dgpu-linux-amd64:latest
"$VENV/bin/monai-deploy" package my_app -c my_app/app.yaml \
    -m "$PWD/models" \
    --monai-deploy-sdk-file "$WHEEL" \
    -t cchmc-tcv-fast:latest --platform x86_64 --cuda 13
```

What's baked: 5× fold weights + `jsonpkls/` at `/opt/holoscan/models/`, app
code at `/opt/holoscan/app`, `nnunetv2==2.8.1` (PyPI), `torch 2.13.0+cu130`,
`monai 1.5.2`, `holoscan-cu13`. Manifest sets `sharedMemory: "4Gi"`.
**Verified image:** sha256 `eb0f12609d85…27db35` (16.4 GB).

---

## 2. Run — batch (one study, one shot)

```bash
monai-deploy run cchmc-tcv-fast-x64-workstation-dgpu-linux-amd64:latest \
    -i <input-dicom-dir> -o <output-dir> \
    --gpus <N> --uid 1000 --gid 10000
```

- `<input-dicom-dir>`: one flat folder of a single study's `.dcm` files.
- `<output-dir>`: must be writable by uid 1000 → `chmod 0777 <output-dir>`.
- Produces `<output-dir>/SEG/*.dcm` + `<output-dir>/SR/*.dcm`.
- No `.npy`/`emit_meta.json` in the MAP. (Re-enable dev emission: add
  `-e HOLOSCAN_EMIT_NPY=1`.)
- **Gotcha:** `monai-deploy run` exits `0` even on app failure. The real
  signal is the log line `Application exited with N`.

---

## 3. Run — serve (persistent daemon, many studies)

`monai-deploy run` has no port-forward flag, so launch serve with raw docker:

```bash
docker run -d --name tcv-serve \
    --gpus '"device=<N>"' --shm-size=4g \
    -v <host-payloads>:/payloads \
    -p 8765:8765 \
    -e MAP_GRPC_PORT=8765 -e MAP_PAYLOADS_ROOT=/payloads \
    -e HOLOSCAN_MODEL_PATH=/opt/holoscan/models \
    --entrypoint bash cchmc-tcv-fast-x64-workstation-dgpu-linux-amd64:latest \
    -c "cd /opt/holoscan/app && python3 __main__.py serve"
```

- Models load **once** (5/0 residency). No per-study reload.
- Wait for readiness: `CheckHealth` returns `INITIALIZING` → `IDLE`.
- **Both** `input_path` and `output_path` must live **under `MAP_PAYLOADS_ROOT`**
  (i.e. under `/payloads` in the container).
- RPC `study_uid` must equal the on-disk DICOM **`StudyID`** (the short
  numeric tag), not the long `StudyInstanceUID`.
- Tear down: `docker stop tcv-serve && docker rm tcv-serve`.

---

## 4. Drive the daemon — folder-pointing client

```bash
"$VENV/bin/python" scripts/serve_folder_client.py \
    --folder <dir-with-studies> --staging <staging-dir> --port 8765
```

- `<dir-with-studies>`: `<folder>/<name>/series1/*.dcm` (≥ 2 levels; deeper
  nesting searched, non-`.dcm` skipped).
- Prints `DISCOVERED …`, one `STUDY <id> … OK latency=…ms stages=…` per study,
  then a final JSON summary. **Exit 0 only if every study succeeded.**
- Outputs land under `<staging>/out_<uid-slug>/{SEG,SR}`.
- Worked example: 2 studies → `{"ok": 2, "failed": 0}`.

---

## 5. Output contract

| File | SOP class | Content |
|---|---|---|
| **SEG** | `1.2.840.10008.5.1.4.1.1.66.4` | 3 bitplane segments (heart / l_lung / r_lung, SCT codes), `NumberOfFrames = 3 × slices`, original CT geometry |
| **SR** | `1.2.840.10008.5.1.4.1.1.88.11` | exactly 3 lines: `Heart Volume: <v> mL`, `Left lung Volume: <v> mL`, `Right lung Volume: <v> mL` (v = voxels × |det(affine)|/1000, 2 dp) |
| *(optional / diagnostic)* | — | `seg_tcv_dhw.npy` (uint8 DHW) + `emit_meta.json`, gated by `HOLOSCAN_EMIT_NPY` (off by default in the MAP) |

---

## 6. Serve contract (what the daemon guarantees)

- **CheckHealth:** `INITIALIZING → IDLE(models[0]={config 3d_fullres,
  trainer nnUNetTrainer_250epochs_NoMirroring, folds 5, patch [256,256,256]},
  queue_depth/capacity, in_flight_study_uid) → BUSY → IDLE`.
- **ProcessStudy** (paths-only `study_uid` + `input_path` + `output_path`):
  bounded FIFO (capacity 4). Typed rejections: `INVALID_ARGUMENT`,
  `ALREADY_EXISTS` (same uid in-flight/queued), `RESOURCE_EXHAUSTED`
  (queue full / scratch space), `INTERNAL` + `stage_at_failure`.
  Response carries `stages[]` + `latency_ms`.
- **Scratch:** per-study private dir (`$MAP_SCRATCH_ROOT`, default a 0700
  tempdir; point it at a `--shm-size` tmpfs mount for RAM-backed scratch);
  deterministic purge on success, fault, and rejection (RESIDUE = 0).
- **Kill safety:** SIGKILL mid-write → **0 files at the final output path**
  (atomic staging-then-rename at egress); re-submitting the killed uid
  republishes idempotently after a restart.

---

## 7. Performance — batch vs serve (measured)

Batch pays a fixed per-run cost (container start + 5-fold model load, ~20–30 s
on the reference GPU) on top of inference; serve pays it once at daemon boot
and each study is pure pipeline. Measured end-to-end per study
(`3d_fullres` 5-fold ensemble):

| Study (slices) | batch wall (GPU host) | serve RPC (GPU host) | serve RPC (baked image)¹ |
|---|---|---|---|
| Study 1 (160) | 52 s | 22.6 s | 41.4 s |
| Study 2 (480) | 117 s | 85.1 s | 109.8 s² |

¹ The baked-image run used a GPU with ~10 GiB of unrelated baseline load;
GPU-host rows share one idle GPU and are the cleaner comparison.
² Post-kill re-publish on the restarted daemon (same daemon lifetime, no
reload — residency 5/0).

Rule of thumb: one study → batch is fine; more than a couple of studies →
serve (first study amortizes the load, the rest run pipeline-only).

---

## 8. Verification (4-study reference corpus)

- **Parity vs pinned pythonic nnUNetv2 oracle** — 12/12 per-class IoU cells
  ≥ 0.99. Worst cell 0.999840. Repeat run byte-identical (deterministic).

| Study | heart | l_lung | r_lung |
|---|---|---|---|
| Study 1 (160 slices) | 0.999947 | 0.999938 | 0.999946 |
| Study 2 (480 slices) | 0.999967 | 0.999955 | 0.999840 |
| Study 3 (192 slices) | 0.999964 | 0.999968 | 0.999962 |
| Study 4 (417 slices) | 0.999969 | 0.999961 | 0.999976 |

- **In-container E2E on the baked image** — 22/22 PASS: batch
  (exit-0, baked-weights, contract, zero-npy, SEG byte-identical to the host
  batch baseline) + serve (lifecycle, `ALREADY_EXISTS`,
  0 final files, 4 GiB shm zero-residue, 5/0 residency, clean teardown, GPU
  released).

---

## 9. Caveats

- **Corpus:** 4 reference studies (Study 1–4); generalization beyond them
  is unverified here.
- **Stack:** build venv and image are near-identical (monai 1.5.2, numpy 2.2.6,
  pydicom 3.0.2, nnunetv2 2.8.1, torch 2.13.0); the image's torch is `+cu130`
  and its `holoscan-cu13` is ≥ 4.3.0 — pixel-identical in-container.
- **SR sha256 is run-volatile** (timestamps/internal UIDs). The bar is
  content-level parity, not file hash.

---

## 10. Development notes

- **Build venv:** create a dedicated venv with the editable SDK install
  (`pip install -e <repo-root>`); never use the system Python.
- **`nnunetv2`:** the vendored editable copy under the repo root (`nnUNet/`)
  is the canonical source; the MAP image pins PyPI `nnunetv2==2.8.1`.
- **Model weights:** 5.3 GB; downloaded via `scripts/download_models.sh`
  (Zenodo); stored under `models/` (gitignored); verified against
  `MANIFEST.sha256`.