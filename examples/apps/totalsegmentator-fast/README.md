# totalsegmentator-fast

## What this is

A GPU-resident, nnU-Net-based port of [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
(TS 2.18) as a [MONAI Deploy](https://github.com/Project-MONAI/MONAI-Deploy) application: a CT DICOM
study goes in; numpy segmentation arrays plus one DICOM-SEG and one DICOM-SR series come out.
A task-generic registry covers **all 45 registry entries** (43 TS 2.18 catalog tasks + 2 Phase-26
dev layout-test entries), selected at runtime with the `HOLOSCAN_TASK` environment variable
(default `total`); model weights load exactly once per process and stay resident on the GPU for
the lifetime of the run. The deployable app root is [`my_app/app.yaml`](./my_app/app.yaml);
packaged MAPs for two tasks: [Building a MAP](#building-a-map).

## Quick start

**Prerequisites**

- A single NVIDIA GPU (CUDA 13-capable) — GPU-resident by design (a RAPIDS RMM allocator is
  initialized before the engine starts). Numbers below were measured on an **NVIDIA A100-SXM4-40GB**;
  that model is not required, but expect ≥ 40 GB-class memory for the 5-part `total` task (the
  shipped `app.yaml` declares `gpuMemory: 20Gi` — a MAP default, not a validated minimum).
- Python 3.12: `pip install -r my_app/requirements.txt` (full list: torch 2.13.0+cu130,
  cupy-cuda13x, rmm-cu13, monai 1.5.2, pydicom 3.0.2, highdicom, nibabel, itk, numpy 2.2.6, and
  transitive deps). `nnunetv2` is a normal PyPI dependency pinned `nnunetv2==2.8.1` in the same
  file — no PYTHONPATH or vendored-checkout step. (The repo's `nnUNet/` checkout is dev/operator
  tooling only, not a runtime dependency.)
- Model weights in the bundle layout (below). **No weights are shipped in the repo** — acquire per
  [Model checkpoints](#model-checkpoints--bundle-layout).
- Input: one CT DICOM study — a directory of `.dcm` files.

**Run command** — one task per invocation (default `total`, which internally runs all 5 model parts):

```bash
export CUDA_VISIBLE_DEVICES=0
# Point PYTHONPATH at this example's directory (<repo>/examples/apps/totalsegmentator-fast)
export PYTHONPATH=/path/to/monai-deploy-app-sdk/examples/apps/totalsegmentator-fast
python -m my_app \
  -i /path/to/ct_study_dir \
  -m /path/to/ct-totalsegmentator-map/models/total \
  -o /path/to/output_dir
```

- `-i` — the CT DICOM study folder. · `-m` — the task's model root (`models/<task>/`; fail-fast
  resolved, so a missing model errors immediately). · `-o` — output directory (the app refuses a
  non-empty one, so a rerun never mixes with stale results).

**Environment variables**

| Variable | Default | Meaning |
|---|---|---|
| `HOLOSCAN_TASK` | `total` | Which registry task to run. Unknown values fail fast, naming the valid set. |
| `HOLOSCAN_MODEL_PARTS` | — | Comma-separated part subset for multi-part tasks (e.g. `organs,vertebrae` for `total`). |
| `HOLOSCAN_GPU_METRICS` | `1` | Toggle GPU memory-metric collection. |
| `HOLOSCAN_SCHEDULER_WORKERS` | `5` | Scheduler worker count. |
| `HOLOSCAN_CONCURRENT_FRAGMENTS` | `1` | Max concurrent fragments. |
| `HOLOSCAN_EMIT_NPY` | `1` | Gate for the debug/terminal numpy `.npy` + `emit_meta.json` writes. `1` (or unset) = writes (dev/validation default); `0` = disabled. The shipped MAP images bake `0`. |

**Output contract** (into `-o`) — the DICOM-SEG/SR writers are **never optional**: every run
produces the `SEG/` and `SR/` series regardless of settings. The debug/terminal numpy `.npy`
arrays + `emit_meta.json` are gated by `HOLOSCAN_EMIT_NPY` — default `1` (ON) for dev/validation;
the shipped MAP images bake `0`, so a MAP output dir contains only `SEG/` and `SR/`:

- `-o` root (when `HOLOSCAN_EMIT_NPY` is `1`/unset) — plain numpy `.npy` segmentation arrays
  (uint8 label maps) + `emit_meta.json` (per-array shape/dtype/label histograms):
  - `total` (5-part): `seg_total_dhw.npy` + `seg_total_sar.npy` (both exact study shape).
  - `body` (and single-part tasks): `seg_<part>_modelspace.npy` (1.5 mm model-resolution space) +
    `seg_<part>_dicom.npy` (original-DICOM orientation) — e.g. `seg_body_modelspace.npy` /
    `seg_body_dicom.npy`.
- `SEG/` — exactly one DICOM-SEG series; `SeriesDescription` = "AI Generated DICOM SEG; Not for
  Clinical Use."
- `SR/` — exactly one DICOM-SR series; `SeriesDescription` = "AI Generated DICOM SR; Not for
  Clinical Use." (text SR: "Textual report from AI algorithm. Not for clinical use.")
- Unlike the upstream TotalSegmentator 2.18 CLI, which writes NIfTI files, this app never produces
  NIfTI — the `.npy` arrays are plain numpy (the DICOM-SEG/SR series are the DICOM deliverables).

## How it works (families of tasks)

The registry (`my_app/task_specs/`) describes each task as pure data (parts, label offsets, crop
spec, resampling, label tables); the app builds the correct DAG from that description at startup —
one code path serves all 45 registry entries. Three pipeline families:

- **S — single-model.** One nnU-Net bundle segments the (pre-resampled) whole volume directly.
- **M — multi-part.** Several nnU-Net bundles run (e.g. `total` runs organs / vertebrae / cardiac /
  muscles / ribs); label maps merge in parts order (later parts win) with per-part label offsets —
  `total` produces a single 117-class merged segmentation.
- **C — crop pre-stage.** A coarse whole-body model (typically the 6 mm or 3 mm `total` model) runs
  first; its label map yields a crop mask over the target organs, the volume is cropped (20 mm
  per-axis padding, native resolution), the main model runs on the crop, and the result is pasted
  back. C composes with S or M (e.g. `headneck_muscles` = crop + 2-part main model, `C+M`).

## Model checkpoints & bundle layout

The repo ships **no weights** (the `ct-totalsegmentator-map/models/` tree is git-ignored). Three
acquisition paths:

1. **`pip install totalsegmentator` auto-download** — running the package (e.g.
   `python -c "import totalsegmentator; totalsegmentator(input_path='...', output_path='...', task='total')"`)
   triggers `download_pretrained_weights` on first use; weights land in
   `~/.totalsegmentator/nnunet/results/<foldername>` (honors `TOTALSEG_WEIGHTS_PATH`).
2. **Official TS model server, by task_id.** The **working** import in TS 2.18.0:

   ```python
   from totalsegmentator.libs import download_pretrained_weights
   download_pretrained_weights(<task_id>)   # e.g. 299 for `body`
   ```

   **Pitfall:** the top-level `from totalsegmentator import download_pretrained_weights` wrapper is
   broken in 2.18.0 (its inner `from libs import …` fails) — always import from
   `totalsegmentator.libs` directly. The server fetches
   `https://github.com/wasserth/TotalSegmentator/releases/download/{version_tag}/{foldername}.zip`.
3. **The [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) GitHub project** — release
   assets / `v*-weights` tags; manual download for licensed or unreleased tasks (e.g. `total_v3`'s
   `v3.0.0-weights` release is still unpublished — see [Caveats & fidelity](#caveats--fidelity)).

**Bundle layout.** Bundle root `ct-totalsegmentator-map/` at `<repo-root>/ct-totalsegmentator-map/`
(a sibling of `examples/`); `-m` points into it (e.g. `<repo>/ct-totalsegmentator-map/models/total`);
every `models/<task>/<part>/` path below is relative to that root. Each part bundle must contain:

```
models/<task>/<part>/
├── jsonpkls/
│   ├── plans.json                  (required — the nnU-Net plans file TS checkpoints carry)
│   ├── dataset.json                (required)
│   └── postprocessing.pkl          (optional)
└── <config_name>/                  # 3d_fullres | 3d_fullres_high | 3d_lowres_high
    ├── nnunet_checkpoint.pth       (optional metadata sidecar)
    └── fold_0/
        └── final_model.pt          (required; best_model.pt / model.pt also accepted)
```

Checkpoint preference: `final_model.pt` → `best_model.pt` → `model.pt`. Concrete examples: `body`
(single part) at `models/body/body/…`; `liver_segments` (crop cascade, 2 parts) at
`models/liver_segments/total_6mm/` (crop model) + `models/liver_segments/ct_liver_segments/`
(post-crop main). Root resolution is fail-fast: `-m` may point at `models/`, `models/<task>/`, or a
bare task dir; missing `jsonpkls/plans.json` or per-part weights produce an explicit error.

**Converting a TS download.** Use the **lower-level** converter
`app_total.nnunet_bundle.convert_nnunet_to_monai_bundle(nnunet_config, bundle_root_folder, fold=0,
checkpoint_type="final")` — NOT the top-level `convert_nnunet` CLI, which hard-requires
`inference_information.json` (an artifact TS weight zips lack) and fails `FileNotFoundError` on
them. Run it with `nnUNet_results` pointing at the directory **containing** the dataset folders
(parent of `Dataset<id>_<name>/<trainer>__<plans>__<config>/`, e.g.
`~/.totalsegmentator/nnunet/results`); it writes `<config>/fold_0/final_model.pt` +
`nnunet_checkpoint.pth` and copies `jsonpkls/{plans.json,dataset.json}`, which you then move into
`models/<task>/<part>/`.

## Adapting to your own nnUNet bundle

To run this app on bundles that are not the shipped task weights, you need: a model you can
acquire, the layout contract below, and a registry entry (`HOLOSCAN_TASK`) that references it.

**Get your bundle.** The three acquisition paths in
[Model checkpoints & bundle layout](#model-checkpoints--bundle-layout) apply unchanged — pip
auto-download, the official TS model server (`download_pretrained_weights(<task_id>)`), the
TotalSegmentator GitHub release assets, and `convert_nnunet_to_monai_bundle` (documented above)
for an arbitrary nnUNet bundle. The app resolves the model root at startup and **fails fast**:
any missing `jsonpkls/plans.json`, part dir, or checkpoint is named in the error — no silent
fallback (no implicit `3d_fullres` substitution for unknown part→config mappings since v3.3).

**The layout contract.** Each part bundle must look exactly like this:

```
models/<task>/<part>/
├── jsonpkls/
│   ├── plans.json              (required)
│   ├── dataset.json            (required)
│   └── postprocessing.pkl      (optional)
└── <config_name>/fold_N/       # any fold_N dirs present are all loaded and averaged
    ├── final_model.pt          (preferred; best_model.pt / model.pt accepted)
    └── nnunet_checkpoint.pth   (optional metadata sidecar)
```

Two hardening notes: the completeness check requires `fold_0/final_model.pt` per part (a bundle
shipping only non-zero folds is rejected at root resolution even though the loaders would read
them — Phase 25 finding CFG02-01, fixed in Phase 26), and the part→config mapping must be in the
registry entry (unknown mappings error instead of falling back — CFG02-02). Multi-fold: any
number of `fold_N` dirs are all loaded and averaged (verified at 5 folds — [Config support
matrix](#config-support-matrix) below).

**Run it.** Dev-run command shape — the exact shape the Phase 26 gate matrix executed (22/22 PASS):

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/monai-deploy-app-sdk/examples/apps/totalsegmentator-fast
export HOLOSCAN_TASK=<your_task>          # registry task name; unknown values fail fast
python -m my_app \
  -i /path/to/ct_study_dir \
  -m /path/to/bundle_root \
  -o /path/to/fresh_empty_output_dir
```

Env knobs a bundle adaptor actually needs: `HOLOSCAN_MODEL_PARTS` (comma-separated part subset
for multi-part tasks) and `HOLOSCAN_EMIT_NPY` (dev `1` / MAP-baked `0`). For a packaged MAP:
see [Building a MAP](#building-a-map).

> **Source:** command shape reflects the Phase 26 gate-matrix runs — 22/22 cells PASS
> (`.planning/phases/26-config-matrix-fold-count-hyperparameter-gate-verification/26-gate-runs.csv`, runner `.planning/scripts/p26_run_cell.sh`). No invented flags.

**What gets produced.** The Option-A contract is identical to the shipped tasks: with the dev
default (`HOLOSCAN_EMIT_NPY` `1`/unset) the `-o` root also receives plain-numpy `.npy` label maps
(per-task names follow the registry task, e.g. `seg_<part>_modelspace.npy` / `seg_<part>_dicom.npy`,
or `seg_total_dhw.npy` + `seg_total_sar.npy`) plus `emit_meta.json`; every run always also writes
one DICOM-SEG (`SEG/`) and one DICOM-SR (`SR/`). Full contract: [Output contract](#output-contract-into--o).

## Config support matrix

Support status per nnUNet config layout × fold count. Status vocabulary (strict):
**verified** = explicitly exercised and asserted in a Phase 26 gate cell (exactly the 22/22
cells + 4/4 pairs in `26-gate-report.json`, each citing committed evidence below);
**inferred-only** = code-generic / honored per Phase 25 evidence but NOT varied in Phase 26
(each cell states what would verify it); **pending** = not runnable today. Do not over-read
"verified" — caveat below.

| Layout | 1-fold | 5-fold | Notes + evidence |
|---|---|---|---|
| `3d_fullres` | **verified** | **verified** | Real weights, unpatched plans; 33294 npy byte-identical (sha256) to the untouched registry-bundle run (copy fidelity). Evidence: `26-gate_fullres_only_{33294,31322,64199}.json`; fold5 pair: `26-gate_pair_fold5_{33294,31322,64199}.json` |
| `3d_lowres` | **verified** | **inferred-only** | Weights-reused, layout-exercised (meta-patched `init_args.configuration`). `26-gate_lowres_only_{33294,31322,64199}.json`. 5-fold not run — fold machinery is layout-agnostic; a 5-fold lowres bundle (Phase 26 committed builder) would verify it |
| `3d_cascade_fullres` | **verified** | **inferred-only** | Weights-reused, layout-exercised (zero one-hot cascade input, `HOLOSCAN_SYNTHETIC_CASCADE=1`); documented fg≥0 exception. `26-gate_cascade_only_{33294,31322,64199}.json`. 5-fold: same rationale as lowres |
| `3d_fullres` + `3d_lowres` (combo bundle) | **verified** | **inferred-only** | Per-spec config selection from one bundle: `total`→3d_fullres and `organs_lowres_test`→3d_lowres each with fg identical to their single-layout cells across 6 double-runs. `26-gate_fullres_lowres_total_*.json` / `26-gate_fullres_lowres_lowres_*.json`. 5-fold: same rationale |
| `2d` / `2d_u` | **pending** | **pending** | See the 2d paragraph below |

**Fold5 honesty caveat.** (All cited files under `.planning/phases/26-config-matrix-fold-count-hyperparameter-gate-verification/`.) The fold5 bundle replicates `fold_0`'s weights ×5 (registry bundles ship fold_0 only), so "verified" proves the fold-count/accumulation *mechanism* — 5× load, FP32 accumulate, ÷5 — with 100 % argmax label-map agreement vs 1-fold (0 flip voxels, 3 studies), +5 s/+5 s/+2 s wall, in-graph inference span ×4.0 — not real-ensemble quality. Likewise the lowres/cascade "verified" cells copy `3d_fullres` weights into the layout: the layout is exercised end-to-end, not lowres/cascade model quality.

**2d is pending verification.** The pipeline wiring is config-generic (Phase 25 inventory rows
`2d` / `2d_u`), but no 2d model weights exist for any shipped task and the inference pipeline is
3D-only today (preprocessor dimension asserts; 5-D swin tensors). A real 2d model run is deferred
to the separate 2d nnUNet app. **Do not point a 2d bundle at this app.**

### Inference hyperparameters

One row per setting in the Phase 25 inventory — 27 rows, the 1:1 traceability key
(`.planning/phases/25-hyperparameter-inventory-config-audit/25-hyperparameter-inventory.csv`); same
status vocabulary as above; what the app will *never* do for the fixed rows is in the next subsection.

| Setting | Source | Class | Status | Effect / what the app does |
|---|---|---|---|---|
| `spacing_in_mm` (config `spacing`) | plans.json | honored | inferred-only | Drives pre-resample + nnUNet-chain resample (census: 8 distinct 3d tuples). Verify with: a non-corpus spacing tuple |
| `resampling_fn_data_kwargs.order` / `order_z` | plans.json | honored | inferred-only | Data-resampling degree (census `order=3, order_z=0` 22/22). Verify with: a different order |
| `resampling_fn_seg_kwargs.order` / `order_z` | plans.json | honored | inferred-only | Seg-resampling order, cascade one-hot path (census `order=1, order_z=0`). Verify with: a different seg order |
| `force_separate_z` | plans.json | honored | inferred-only | Separate-z resampling (census `null` 22/22 = auto from anisotropy). Verify with: an anisotropic plan flipping the axis |
| `allow_torch_anisotropy` | plans.json | unsupported | **pending** | In no corpus plan and 0 consumers in vendored nnUNet 2.8.1 — nothing to honor today; row flips if a future plans format introduces it |
| `previous_stage` (cascade producer) | plans.json | honored | inferred-only | Drives cascade `lowres_seg` input + producer auto-insertion (exercised by verified cascade cells). Verify as a *hyperparameter* with: a producer other than `3d_lowres` |
| `transpose_forward` / `transpose_backward` | plans.json | honored | inferred-only | Channel-order transposes (census `[0,1,2]` 22/22). Verify with: a non-identity transpose |
| `normalization_schemes` | plans.json | honored | inferred-only | Per-channel ZScore/CT/NoNormalization replicas (census single-channel CT 22/22). Verify with: a ZScore or multi-channel plan |
| `use_mask_for_norm` | plans.json | honored | inferred-only | Masked-norm branch (census `[false]` 22/22). Verify with: a plan with `true` |
| `foreground_intensity_properties_per_channel` | plans.json | honored | inferred-only | CT-clip percentiles/mean/std in the CT branch (present 22/22). Verify with: materially different percentiles |
| `preprocessor_name` | plans.json | fixed (benign) | fixed | Key never read — the app unconditionally replicates the `DefaultPreprocessor` chain; all 22 corpus plans use it → zero divergence |
| `border_mode` / `border_value` | plans.json (legacy; no 2.8.1 key) | fixed (benign) | fixed | Resize `mode='edge'` + pad 0 hardcoded = the reference's own hardcodes |
| `patch_size` | plans.json | honored | inferred-only | Sliding-window tile size via `ConfigurationManager.patch_size` (4 distinct 3d patches). Verify with: a patch size outside the census set |
| `step_size` (tile step size) | code-default (not a 2.8.1 key) | fixed (benign) | fixed | Per-spec `TaskSpec.tile_step_size`: 0.8 total family / 0.5 others = the TS `nnunet.py` rule |
| `use_gaussian` | code-default | fixed (benign) | fixed | Hardcoded `True` at all 5 wiring sites = the reference default |
| gaussian blend parameters (`sigma_scale` / `value_scaling_factor`) | code-default | fixed (benign) | fixed | Calls the *reference* `compute_gaussian` with stock defaults (1/8, 10) — divergence structurally impossible |
| `mirror_axes` | checkpoint-meta (`inference_allowed_mirroring_axes`) | honored | **verified** | A/B on a mirror-capable bundle: verbatim log line `TTA on: config=3d_fullres mirror_axes=(0, 1, 2) combinations=7`. `26-gate_pair_mirror_33294.json` |
| `use_mirroring` (TTA on/off) | code-default pre-v3.3 → plans key | honored (was fixed-divergent) | **verified** | AFTER state: plan-driven, honored. A/B: off byte-identical to the unmodified registry run; on = 0.1986 % fg diff + TTA log line. `26-gate_pair_mirror_33294.json`. Phase 25 found it fixed-divergent (hardcoded `False` ×5); Phase 26 resolved it |
| `2d` (config family) | plans.json | unsupported | **pending** | 3D-only pipeline, no 2d dirs shipped — see the 2d paragraph (CFG-03) |
| `2d_u` (config family) | plans.json | unsupported | **pending** | Same as `2d`; not even in the 22-plan corpus (CFG-03) |
| fold count / ensemble averaging | checkpoint-meta (`fold_*` dirs) | honored | **verified** | 5-fold pair: 100 % argmax agreement vs 1-fold (0 flip voxels, 3 studies), FP32 accumulate ÷fold-count. `26-gate_pair_fold5_{33294,31322,64199}.json`. Caveat: weights-reused — machinery, not ensemble quality |
| model list / config selection | plans.json | honored | **verified** | Per-spec selection proven by the 6 fullres_lowres double-runs (fg identical per study to the single-layout cells) |
| `postprocessing.pkl` rules | bundle jsonpkls | honored | inferred-only | All 15 registry bundles ship no pkl → live path is the identity no-op (matches reference). Verify with: a bundle shipping a pkl with rules |
| `labels` (label table) | dataset.json | honored | inferred-only | Loaded into `PreprocessParams.labels`/`foreground_labels` (10 table sizes in census). Verify with: a structurally different label table |
| `label_manager` | plans.json | honored | inferred-only | Via the reference's own `PlansManager`/`LabelManager` (census `LabelManager` 22/22). Verify with: a non-LabelManager plans format |
| `num_channels` / `channel_names` | dataset.json | honored | inferred-only | Reference's own `determine_num_input_channels` imported and called (no re-implementation drift). Verify with: a multi-channel dataset |
| network architecture (`UNet_class_name`/`architecture` + init kwargs) | plans.json | honored | inferred-only | Network built from `ConfigurationManager`-resolved kwargs via the reference builder (census `PlainConvUNet` 22/22). Verify with: another architecture block |

### Fixed & unsupported settings (what the app will never do)

The 9 fixed/unsupported inventory rows, 1:1 with `25-hyperparameter-inventory.csv` (Phase-26
dispositions per `26-GATE-REPORT.md` §7):

1. **`preprocessor_name` — fixed, benign.** Will *never* read the key: unconditionally replicates
   the `DefaultPreprocessor` chain; confirmed fixed (all 22 corpus plans use it → zero divergence).
2. **`border_mode` / `border_value` — fixed, benign.** No 2.8.1 plans key exists; hardcodes
   resize `mode='edge'` + pad 0 = the reference's own hardcodes. Never honors an alternate mode.
3. **`step_size` — fixed, benign.** Not plan-configurable in 2.8.1; pinned per spec
   (`TaskSpec.tile_step_size` 0.8 total family / 0.5 others = the TS `nnunet.py` rule). Never
   read from a plan.
4. **`use_gaussian` — fixed, benign.** Hardcoded `True` at all 5 wiring sites (reference
   default). Never runs without the gaussian blend.
5. **gaussian blend parameters (`sigma_scale` / `value_scaling_factor`) — fixed, benign.** Calls
   the *reference* `compute_gaussian` with stock defaults; divergence structurally impossible.
6. **`use_mirroring` — fixed-divergent in Phase 25, RESOLVED in Phase 26.** Will *never* again
   ignore a plan's `use_mirroring` / `inference_allowed_mirroring_axes` — now plan-driven and
   honored (A/B-verified: off byte-identical to the unmodified registry run, on = 0.1986 % fg
   diff + `combinations=7` log line; `26-gate_pair_mirror_33294.json`). History: Phase 25 found all
   5 wiring sites hardcoded `False` (NoMirroring oracle bundles); Phase 26 made it plan-driven —
   OFF by default, ON when the plan says so.
7. **`allow_torch_anisotropy` — unsupported (pending).** In no corpus plan and 0 consumers in
   vendored 2.8.1 — nothing to honor today; never read unless a future plans format introduces it.
8. **`2d` — unsupported (pending).** 3D-only pipeline (preprocessor dimension asserts); no 2d
   weights shipped for any task; deferred to the separate 2d nnUNet app (CFG-03).
9. **`2d_u` — unsupported (pending).** Same as `2d`; not even present in the corpus. Never runs
   2d_u.

**Plus (task-spec data, not a plans key):** `TaskSpec.tta` (all tasks `tta=False`, never read)
is **DORMANT / SUPERSEDED** by the plan-driven `use_mirroring` key — deliberately not wired.

## Building a MAP

Linux x86_64 CUDA-13 MAP containers: **two separate images, one per task** (`total` 5-part and
`body` single-model) — never a combined image. The chosen task's weights are baked in at build
time; `nnunetv2` is pip-installed from PyPI (pinned `nnunetv2==2.8.1`) during the build, so the
image is self-contained.

**Prerequisite:** the `monai-deploy` CLI — install the SDK (latest) from PyPI:

```bash
pip install monai-deploy-app-sdk
# or directly from the GitHub repo:
pip install "git+https://github.com/Project-MONAI/MONAI-Deploy-SDK.git"
```

Then make the weight trees world-readable (baked as root; the app runs as uid 1000):

```bash
chmod -R a+rX <repo>/ct-totalsegmentator-map/models/total <repo>/ct-totalsegmentator-map/models/body
```

From `examples/apps/totalsegmentator-fast/`, build both MAPs (verified — `total` 596 M bake →
~11.3 GB image; `body` 120 M bake → ~10.8 GB image):

```bash
# total MAP
monai-deploy package my_app -c my_app/app.yaml \
    -m <repo>/ct-totalsegmentator-map/models/total \
    -t totalsegmentator-fast-total:latest \
    --platform x86_64 --cuda 13
# packager appends -x64-workstation-dgpu-linux-amd64 -> final tag
#   totalsegmentator-fast-total-x64-workstation-dgpu-linux-amd64:latest

# body MAP (same command, body weights)
monai-deploy package my_app -c my_app/app.yaml \
    -m <repo>/ct-totalsegmentator-map/models/body \
    -t totalsegmentator-fast-body:latest \
    --platform x86_64 --cuda 13
# -> totalsegmentator-fast-body-x64-workstation-dgpu-linux-amd64:latest
```

To build against a specific local SDK wheel, add `--monai-deploy-sdk-file <wheel>` (the wheel
filename must be a valid PEP 440 name, or in-image pip fails on it).

CLI notes (learned the hard way):

- **Short `-t` tag** — the packager appends the platform suffix; passing the full
  `…-x64-workstation-dgpu-linux-amd64` tag produces a doubled suffix.
- **`-c my_app/app.yaml`** — the CLI resolves `-c` against the CWD, and the manifest lives inside
  the application dir.

**Both images need one extra step**: holoscan-cli 4.2 never forwards the app.yaml `environment`
section (or a run-time `--config` file's) to the container env — env comes only from the
image-embedded `/etc/holoscan/app.json`, which the packager fills with fixed defaults. So each
image gets its committed manifest baked in via one more docker layer: the **total** manifest
bakes `HOLOSCAN_EMIT_NPY=0` (MAP output = DICOM-SEG/SR only; the numpy emit stays ON for local
dev), and the **body** manifest bakes `HOLOSCAN_TASK=body` + `HOLOSCAN_EMIT_NPY=0`:

```bash
# total MAP manifest patch
mkdir -p /tmp/p24_total_manifest_patch
cp examples/apps/totalsegmentator-fast/map-config/app.total.json /tmp/p24_total_manifest_patch/app.json
printf 'FROM totalsegmentator-fast-total-x64-workstation-dgpu-linux-amd64:latest\nCOPY app.json /etc/holoscan/app.json\n' > /tmp/p24_total_manifest_patch/Dockerfile
docker build -t totalsegmentator-fast-total-x64-workstation-dgpu-linux-amd64:latest /tmp/p24_total_manifest_patch

# body MAP manifest patch
mkdir -p /tmp/p24_body_manifest_patch
cp examples/apps/totalsegmentator-fast/map-config/app.body.json /tmp/p24_body_manifest_patch/app.json
printf 'FROM totalsegmentator-fast-body-x64-workstation-dgpu-linux-amd64:latest\nCOPY app.json /etc/holoscan/app.json\n' > /tmp/p24_body_manifest_patch/Dockerfile
docker build -t totalsegmentator-fast-body-x64-workstation-dgpu-linux-amd64:latest /tmp/p24_body_manifest_patch
```

### Run

Verified run commands (in-container exit 0: `total` on study 64199 in ~51 s; `body` on study 44238
in ~14 s — A100, fresh process incl. container start + model load):

```bash
# total MAP
monai-deploy run totalsegmentator-fast-total-x64-workstation-dgpu-linux-amd64:latest \
  -i /path/to/ct_study_dir \
  -o /path/to/fresh_empty_output_dir --gpus <0-3> --uid 1000 --gid 10000

# body MAP (--config = the committed run config, still accepted by the runner)
monai-deploy run totalsegmentator-fast-body-x64-workstation-dgpu-linux-amd64:latest \
  -i /path/to/ct_study_dir \
  -o /path/to/fresh_empty_output_dir --gpus <0-3> --uid 1000 --gid 10000 \
  --config /path/to/repo/examples/apps/totalsegmentator-fast/map-config/app.body.json
```

Run gotchas:

- `--uid 1000 --gid 10000` are **required** (the in-image `holoscan` user); the CLI default host
  uid fails at import and on output permissions. Host output dirs must be writable by uid 1000
  (`chmod 0777` the run dir) or the writers die `PermissionError`.
- The app refuses a non-empty `-o` (a rerun never mixes with stale results).
- **Exit status:** `monai-deploy run` always returns 0 — read the app log's
  `Application exited with N` line for the true app exit code.
- `--gpus` pins the visible GPU set; pick idle GPUs.

### Outputs (per task)

The shipped MAP images run with `HOLOSCAN_EMIT_NPY=0`, so **MAP output = `SEG/` + `SR/` only**.
With the dev default (env unset/`1`), the numpy emits are added under the `-o` root:

- `total` → MAP: `SEG/` one DICOM-SEG series (117 segments) + `SR/` one DICOM-SR series; dev
  default additionally: `seg_total_dhw.npy` + `seg_total_sar.npy` (exact study shape, uint8,
  labels 0..117) + `emit_meta.json`.
- `body` → MAP: `SEG/` one DICOM-SEG series per non-background segment + `SR/` one DICOM-SR
  series per segment; dev default additionally: `seg_body_modelspace.npy` + `seg_body_dicom.npy`
  (exact study shape, uint8, labels ≤ 2) + `emit_meta.json`.
- Full contract (incl. plain-numpy / no-NIfTI): see [Output contract](#output-contract-into--o).

## Supported tasks (45)

The registry (`my_app/task_specs/__init__.py`, `TASK_REGISTRY`) ships one entry per task = the
TotalSegmentator 2.18 `default` sub-mode. **S = single-model · M = multi-part · C = crop pre-stage**
(composes with S or M, e.g. `C+M`). **Labels** = non-background labels from the registry labels
table; **resample** = task-resolution input pre-resample in mm (`None` = no TS input pre-resample);
**model config** = per-part `config_name`, crop-model part listed first for C families;
**Verified** = agreement validated against the TotalSegmentator 2.18 package itself.

| # | Task | Family | task_id(s) | resample (mm) | model config (parts) | Labels | Licensed | Verified |
|---|---|---|---|---|---|---|---|---|
| 1 | `total` | M (5 parts) | 291–295 | 1.5 | 3d_fullres ×5 (organs/vertebrae/cardiac/muscles/ribs) | 117 | no | ✓ (full corpus) |
| 2 | `total_v3` | M (5 parts) | 831–835 | 1.5 | 3d_fullres ×5 | 117 | no | — (weights unpublished) |
| 3 | `body` | S | 299 | 1.5 | 3d_fullres | 2 | no | ✓ |
| 4 | `total_highres_test (dev)` | S | 957 | 0.75/0.75/1 | 3d_fullres_high | 24 | no (dev task) | — |
| 5 | `lung_vessels` | C+S | 117 | 0.703125/0.703125/1 | 3d_fullres (total_3mm crop) + 3d_fullres | 4 | no | — |
| 6 | `lung_vessels_LEGACY` | C+S | 258 | None | 3d_fullres (total_6mm crop) + 3d_fullres | 2 | no | — |
| 7 | `cerebral_bleed` | C+S | 150 | None | 3d_fullres (total_6mm crop) + 3d_fullres | 1 | no | — |
| 8 | `hip_implant` | C+S | 260 | None | 3d_fullres (total_6mm crop) + 3d_fullres | 1 | no | — |
| 9 | `pleural_pericard_effusion` | C+S | 315 | None | 3d_fullres (total_6mm crop) + 3d_fullres | 3 | no | ✓ |
| 10 | `liver_vessels` | C+S | 8 | None | 3d_fullres (total_6mm crop) + 3d_fullres | 2 | no | — |
| 11 | `head_glands_cavities` | C+S | 775 | 0.75/0.75/1 | 3d_fullres (total_6mm crop) + 3d_fullres_high | 19 | no | — |
| 12 | `headneck_bones_vessels` | C+S | 776 | 0.75/0.75/1 | 3d_fullres (total_6mm crop) + 3d_fullres_high | 12 | no | — |
| 13 | `head_muscles` | C+S | 777 | 0.75/0.75/1 | 3d_fullres (total_6mm crop) + 3d_fullres_high | 11 | no | — |
| 14 | `headneck_muscles` | C+M (3 parts) | 778–779 | 0.75/0.75/1 | 3d_fullres (total_6mm crop) + 3d_fullres_high ×2 (part1/part2) | 23 | no | ✓ |
| 15 | `oculomotor_muscles` | C+S | 351 | 0.472516/0.472516/0.85 | 3d_fullres (total_6mm crop) + 3d_fullres | 19 | no | — |
| 16 | `lung_nodules` | C+S | 913 | 1.5/1.5/1.5 | 3d_fullres (total_6mm crop) + 3d_fullres | 2 | no | — |
| 17 | `kidney_cysts` | C+S | 789 | 1.5/1.5/1.5 | 3d_fullres (total_6mm crop) + 3d_fullres | 2 | no | — |
| 18 | `breasts` | S | 527 | 1.5/1.5/1.5 | 3d_fullres | 1 | no | — |
| 19 | `ventricle_parts` | C+S | 552 | 0.438477/0.43457/1 | 3d_fullres (total_6mm crop) + 3d_fullres | 12 | no | — |
| 20 | `liver_segments` | C+S | 570 | 0.804688/0.804688/1.5 | 3d_fullres (total_6mm crop) + 3d_fullres (ct_liver_segments) | 8 | no | ✓ |
| 21 | `liver_lesions` | C+S | 591 | 0.75/0.75/1 | 3d_fullres (total_3mm crop) + 3d_fullres_high | 1 | no | — |
| 22 | `craniofacial_structures` | C+S | 115 | 0.5/0.5/0.5 | 3d_fullres (total_6mm crop) + 3d_fullres | 7 | no | — |
| 23 | `abdominal_muscles` | C+S | 952 | 0.75/0.75/1 | 3d_fullres (body crop model) + 3d_fullres_high | 22 | no | — |
| 24 | `teeth` | C+S | 113 | 0.5/0.5/0.5 | 3d_fullres (**named crop model** `craniofacial_structures`) + 3d_lowres_high | 77 | no | — |
| 25 | `trunk_cavities` | S | 343 | 1.5/1.5/1.5 | 3d_fullres | 4 | no | ✓ |
| 26 | `vertebrae_body` | S | 305 | 1.5 | 3d_fullres | 2 | no | — |
| 27 | `vertebrae_pp` | S | 803 | 1.5 | 3d_fullres | 24 | no | ✓ (one documented label exception) |
| 28 | `vertebrae_pp_refined` | S | 803 | 1.5 | 3d_fullres | 24 | no | — |
| 29 | `heartchambers_highres` | C+S | 301 | None | 3d_fullres (total_6mm crop) + 3d_fullres | 7 | **YES** | — |
| 30 | `appendicular_bones` | S | 304 | 1.5 | 3d_fullres | 11 | **YES** | — |
| 31 | `tissue_types` | S | 481 | 1.5 | 3d_fullres | 3 | **YES** | — |
| 32 | `tissue_4_types` | S | 485 | 1.5 | 3d_fullres | 4 | **YES** | — |
| 33 | `face` | S | 303 | 1.5 | 3d_fullres | 1 | **YES** | — |
| 34 | `brain_structures` | C+S | 409 | 0.5/0.5/1 | 3d_fullres (total_6mm crop) + 3d_fullres_high | 16 | **YES** | — |
| 35 | `thigh_shoulder_muscles` | S | 857 | 1.5 | 3d_fullres | 18 | **YES** | — |
| 36 | `coronary_arteries` | C+S | 509 | 0.7/0.7/0.7 | 3d_fullres (total_6mm crop) + 3d_fullres_high | 1 | **YES** | — |
| 37 | `coronary_arteries_LEGACY` | C+S | 507 | 0.7/0.7/0.7 | 3d_fullres (total_6mm crop) + 3d_fullres_high | 1 | **YES** | — |
| 38 | `aortic_sinuses` | C+S | 920 | 0.7/0.7/0.7 | 3d_fullres (total_6mm crop) + 3d_fullres_high | 4 | **YES** | — |
| 39 | `renal_arteries` | S | 710 | 1.5 | 3d_fullres | 3 | **YES** | — |
| 40 | `aorta_annulus` | S | 713 | 0.8/0.8/0.8 | 3d_fullres_high | 2 | **YES** | — |
| 41 | `aortic_dissection` | S | 716 | 0.8/0.8/0.8 | 3d_fullres_high | 2 | **YES** | — |
| 42 | `pulmonary_artery_landmarks` | S | 514 | 0.780273/0.780273/1 | 3d_fullres | 7 | **YES** | — |
| 43 | `test (dev)` | C+S | 517 | None | 3d_fullres (total_6mm crop) + 3d_fullres | 1 | no (**dev task**) | — |
| 44 | `organs_lowres_test (dev/layout-test)` | S | 291 | None | 3d_lowres | 24 | no (**dev task**) | — |
| 45 | `organs_cascade_test (dev/layout-test)` | S | 291 | None | 3d_cascade_fullres | 24 | no (**dev task**) | — |

Notes:

- **Verified column.** Seven tasks were validated label-by-label against a fresh TotalSegmentator
  2.18 package run on the same studies: `total` on the full 359-study CT corpus (349/359 studies,
  97.2%, pass the volume-tier agreement bar; pixel identity ≥ 99.986 % on all 359), and `body`,
  `liver_segments`, `headneck_muscles`, `pleural_pericard_effusion`, `trunk_cavities`,
  `vertebrae_pp` each on 3 per-study comparisons (all labels at IoU 1.0000 except one documented
  single-label exception in `vertebrae_pp` — see Caveats). The other 38 entries are not yet
  compared: 37 are complete and runnable (including the four dev/layout-test entries);
  `total_v3` is blocked by unpublished upstream weights.
- **The 14 licensed tasks** (rows 29–42) are **registry metadata only**: running them requires your
  own commercial [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) license and
  checkpoints. None are shipped or pre-downloaded.
- `test` (517), `total_highres_test` (957), `organs_lowres_test`, and `organs_cascade_test` are
  **dev / layout-test tasks**, not supported product tasks — the last two exist to exercise the
  `3d_lowres` and `3d_cascade_fullres` layouts in the Phase 26 gate
  matrix (see [Config support matrix](#config-support-matrix)). Sub-modes: one entry per task =
  the TS 2.18 `default` sub-mode; the `fast`/`fastest` variants (e.g. total 297/298) are catalog
  spacing variants, not registry entries.
- **Extending to other tasks** needs no DAG or pipeline code — the pipeline is fully registry-driven
  (`get_task_spec(task)` builds the topology at startup; entries are generated from the live TS
  2.18 catalog). A new task needs a pure-data `TaskSpec` entry (parts, label offsets, crop spec,
  resampling, trainer, label table, SNOMED table), the model bundle(s) under the task's model root,
  and optionally a SNOMED descriptions module for the DICOM-SR report.

## Performance (vs TotalSegmentator 2.18)

In all 9 benchmark cells the fast app is faster than the TotalSegmentator 2.18 package in
inference-only mode (TS side excludes its per-label `-s` statistics pass), **1.43×–2.16×**.
Measured on a single **NVIDIA A100-SXM4-40GB** per run (fast app on GPU 0, TS 2.18.0 on GPU 1,
**serial** — never two processes on one GPU); medians of 3 measured reps per cell (1 warmup rep
excluded, **zero retries** across all 72 runs).

| Task | Study | Fast app wall s, median (min–max) | TS 2.18 inference-only s, median (min–max) | Speedup (TS/fast) |
|---|---|---|---|---|
| `total` | 44238 (small) | **20** (19–20) | 42 (42–42) | **2.10×** |
| `total` | 64199 (medium) | **46** (45–46) | 66 (66–66) | **1.43×** |
| `total` | 31322 (large) | **73** (72–73) | 158 (156–161) | **2.16×** |
| `liver_segments` | 44238 (small) | **12** (12–12) | 19 (19–19) | **1.58×** |
| `liver_segments` | 64199 (medium) | **27** (27–27) | 49 (49–49) | **1.81×** |
| `liver_segments` | 31322 (large) | **39** (38–39) | 76 (76–76) | **1.95×** |
| `body` | 44238 (small) | **11** (11–12) | 17 (16–17) | **1.55×** |
| `body` | 64199 (medium) | **19** (18–19) | 30 (30–30) | **1.58×** |
| `body` | 31322 (large) | **28** (28–28) | 55 (55–55) | **1.96×** |

**Methodology.** Cells: 3 tasks (`total`, `liver_segments`, `body`) × 3 studies — 44238 (small),
64199 (medium), 31322 (large). Fast app: fresh process per rep; wall = process start→end,
**model load included** (5/2/1 bundles per task) and INCLUDING the output writes (numpy `.npy`
arrays + DICOM-SEG/SR serialization). These benchmark runs used the dev default
(`HOLOSCAN_EMIT_NPY` unset = `1`, numpy emits ON); the shipped MAP images run with emits OFF
(`HOLOSCAN_EMIT_NPY=0`), so MAP deployment is slightly **faster** than these numbers by exactly
the skipped numpy serialization time. TS side: package CLI on
the same studies, GPU-only, **without `-s`/`-sii`**. Timings cover these 3 tasks only — no per-task
timings are claimed for the other 40 catalog tasks. Edge cases: `liver_segments/44238` is the
no-liver skip path (main model never runs); `total/64199` is the slowest-speedup cell (1.43×).

**Inference-only convention.** TS's `-s`/`-sii` flags add a per-label statistics pass that, on
large high-resolution studies, runs single-core and dominates run time (tens of minutes for
`total` on a 157.5M-voxel study, vs ~46 s of actual GPU inference). TS numbers above therefore
exclude only that loop; TS's own CPU pre/post-processing (input resampling, resample-to-original,
NIfTI I/O) is still included.

**Honest framing:** the fast wall is end-to-end (model load + output writes; no NIfTI is ever
produced) — the speedups above are the defensible engine-level number, not an end-to-end-vs-CLI
claim.

**Test data.** Anonymized, de-identified CT studies (TCIA-style IDs such as
`09-15-2004-NA-CT-64199`; the table cites short suffixes 44238 / 64199 / 31322). The full 359-study
corpus served the accuracy comparison of `total` (vs a reference MONAI app and the TS 2.18
package); 4 of those studies covered the stratified per-task comparisons of the 6 verified
single-/multi-part tasks (15 task×study cells, 12 executed — one closed as the documented
single-label exception, 3 `total_v3` cells deferred on unpublished upstream weights). The 3-study
table above is a **timing** benchmark on size-representative studies, not the accuracy validation
set. The corpus is not redistributed with this repo, and its TCIA provenance/availability is not
asserted or verified here.

## Caveats & fidelity

1. **Thin-boundary / small-organ label differences (the documented 151-label `total` exception
   set).** In the full-corpus comparison vs fresh TS 2.18 references, **151 thin-boundary labels**
   fell below per-label IoU 0.99 — per benchmark study: 44238 → 11, 64199 → 71, 31322 → 69; worst
   = **0.8608 Left Rib 5** on 64199 — dominated by ribs/vertebrae/bowel/bladder/cartilages.
   **Accepted, documented cross-engine thin-boundary jitter**: triage showed no app regression
   (max drift vs a previous fast-app build ≤ 0.42 mL), a deterministic pipeline (≤ 0.007 mL), and
   small absolute volume deltas. Sub-mL / thin structures differ most between engines — exactly
   why the agreement bar (item 4) routes them to a volume-delta test.
2. **Documented math-fidelity deviations.** GPU-optimization deviations from the TS 2.18 nnU-Net-fork
   math are allowed only when explicitly documented (where / what / why / quantified impact). The
   two documented deviations (both in `my_app/operators/slidewindow_operator.py`):
   - **FP32 sliding-window / TTA / fold accumulation and FP32 Gaussian kernel, vs the TS fork's
     FP16.** Produces sub-ULP-class logit deltas that flip argmax on a handful of voxels whose
     top-2 logit margin ≈ 0. Why: FP16 `+=` is non-associative (run-to-run nondeterminism), and
     the GPU-residency contract forbids the CPU round-trip the TS fork uses for fold accumulation.
   - **`cudnn.benchmark=False` under RMM.** TS enables `benchmark=True`; the app must force it off
     because torch's benchmark search calls `cacheInfo()`, which is unsupported by the RMM
     pluggable allocator. Impact: per-convolution low-bit weight-summation differences — the same
     jitter class as above.
3. **One documented single-label exception (`vertebrae_pp`).** On study 64199, **label 10
   (vertebrae_T3), IoU 0.987853 < 0.99** (|ΔV| 0.0234 mL on a 1.9038 mL label; 23/24 labels at
   IoU 1.0000). Root-caused to per-process GPU forward-pass kernel-selection non-determinism on
   top-2 logit ties (margins 2.4e-4–2.2e-2); a 9-knob battery, a TS-input swap, 292/292 weight
   bit-identity, and code byte-identity all excluded other causes. Same accepted math-fidelity
   family as item 2; the bar was not loosened and no code was changed.
   **`total_v3`:** the registry entry exists, but the `v3.0.0-weights` release is still
   **unpublished** on the TotalSegmentator GitHub (release list ends at v2.5.0-weights), so no
   reference comparison is possible for it yet.
4. **The volume-tier agreement bar.** **Labels ≥ 1 mL → per-label IoU ≥ 0.99; labels < 1 mL →
   |ΔV| ≤ max(0.2 mL, 1 %).** It separates genuine large-structure agreement from sub-mL
   micro-labels, where volume delta is the honest metric. Full-corpus `total` result on this bar:
   349/359 (97.2%) of studies pass, pixel identity ≥ 99.986 % on all 359.
5. **"Not for Clinical Use" (NfCU) tagging.** Every output series (DICOM-SEG, DICOM-SR, textual
   report) is tagged Not for Clinical Use — see [Output contract](#output-contract-into--o). This
   software is a research/prototyping tool — **not for clinical use**.

## License

This application is part of [MONAI Deploy](https://github.com/Project-MONAI/MONAI-Deploy) and
follows its license. The segmentation pipeline is a port of
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) (AGPL-3.0, with commercial
licensing options — see the upstream repo); model checkpoints are acquired from the upstream
project's official model server and release assets under its terms. The 14
commercial-license-required tasks in the table above additionally require **your own**
TotalSegmentator commercial license and your own checkpoints.

## Not for Clinical Use

This application and all of its outputs (DICOM-SEG / DICOM-SR / textual reports) are a
research and prototyping tool: **not for clinical use**.
