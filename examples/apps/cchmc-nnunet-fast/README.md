# CCHMC nnU-Net Fast App

A MONAI Deploy example app (`cchmc-nnunet-fast`) that replaces the nnU-Net
Python-wrapper approach (`cchmc_nnunet_fifteen_ckpt_app`) with
**Holoscan-native GPU operators**: CT DICOM in, pixel-exact DICOM-SEG/SR/SC
out, with every intermediate step staying on the GPU.

## Status

**Milestone v1.0 complete (2026-08-20).** All four nnU-Net 3D model
configurations (fullres, lowres, cascade, and the default 15-checkpoint
ensemble bundle) run end-to-end through one Holoscan DAG and are
**pixel-exact** against freshly generated reference-app oracles (bundle:
100.00000% SEG byte-identity; SR exact). Measured latency on an
A100-40GB: **104.2 s bundle (1.63× vs the 169.7 s reference app)** and
**49.7 s single fullres-only (3.4×)**.

**Milestone v1.1 — batch-validated + containerized (2026-08-25).** The
pipeline was batch-validated on the 85-study NICU-PULM MR corpus:
**85/85 exit 0** with non-empty SEG/SR/SC, **71.4 s mean** per-study wall
(standalone venv) vs **123.6 s** for the unchanged reference app (≈
**1.7× speedup**). The Linux x86_64 CUDA-13 MAP container (see
**Building a MAP** below) is built with the model baked in and validated on
a 5-study subset (4×256³ + 1×384³): all exit 0, and `SEG` is
**bit-identical** (100.00000% byte-identity, IoU 1.0) to the standalone
run on both shape classes. Evidence:
`.planning/phases/05-batch-validation-map/map_results.md`; the full
per-study three-way comparison is the Phase 6 comparison report
(`.planning/phases/06-comparison-report-docs-merge/`). **v3.4 MAP (Phase 30,
2026-09-08/09):** the MAP was rebuilt on the v3.4 pattern — image `ea556134a5f0`
(12.5 GB) under the same tag, runtime `nnunetv2==2.8.1` from PyPI (no `--add` bake),
in-container verified exit 0 on the 256³ airway study (117 s; see **Building a
MAP** below). The v1.1 image `3fee5ae557e7` above is superseded history.

## Pipeline

```
DICOMSeriesToVolume ──► Preprocess (GPU: CuPy transpose/crop/normalize,
                      │    scipy resample via stock CuPy, single H2D)
                      │
   one Subgraph per model config (HOLOSCAN_MODEL_LIST):
      └─► SlideWindow (GPU inference, TTA, one-shot model load)
            └─► PostResample (GPU resample; cascade configs also emit
                 lowres_seg for the next fragment — zero disk I/O)
                      │
                      ▼
            EnsembleAverage (GPU, list-order reconstruction)
                      │
                      ▼
            Postprocess (CuPy connected components, revert crop/transpose)
                      │
                      ▼
            DICOM-SEG / SR / SC writers (exactly one .cpu() boundary)
```

- **Multi-fragment DAG**: each model config is its own `Subgraph` built by a
  config-generic factory; fragments run **concurrently** under
  `EventBasedScheduler` (independent fragments overlap on separate worker
  threads; per-config `CudaStreamPool`s).
- **Cascade** (`3d_cascade_fullres`): the lowres fragment emits
  `lowres_seg` (argmax, uint8, original orientation) → one-hot float →
  2-channel cascade input. No disk I/O.
- **Memory**: RMM is the torch allocator (imported before holoscan in
  `gpu_bootstrap.py`), initial pool pinned to 4 GiB, pool warmed to the
  `mem_budget.py` budget at compose time; shape-keyed buffer caches
  (`buffer_cache.py`) reuse GPU buffers across studies (zero extra
  cudaMalloc on repeat studies); `3d_lowres` weights are freed after the
  aux fragment finishes (MEM-003).
- **Observability**: per-config NVTX ranges (`preprocess_<cfg>`,
  `inference_<cfg>`, `postresample_<cfg>`) + structured per-study timing
  log (`study_timing_summary`); nsys-ready.

## Supported model configurations

Selected with `HOLOSCAN_MODEL_LIST` (comma-separated). Semantics replicate
the reference `NNUnetSegOperator` exactly (model-list filtering,
lowres-before-cascade reorder, ensemble = run minus `3d_lowres`, and the
reference `ValueError` on an empty ensemble — with a documented
self-ensemble fallback for a single non-auxiliary model).

| `HOLOSCAN_MODEL_LIST` | Behavior |
| --- | --- |
| *(unset)* | Reference default bundle: fullres + lowres + cascade ensembled (15 checkpoints) |
| `3d_fullres` | single fullres model, self-ensemble |
| `3d_lowres` | single lowres model, self-ensemble fallback |
| `3d_cascade_fullres` | auto-expands to `3d_lowres,3d_cascade_fullres` (previous-stage auto-insertion) |
| `3d_fullres,3d_lowres,3d_cascade_fullres` | explicit bundle |

The 2d config wiring is config-generic and unit-verified but is
**blocked on a real 2d model** (none in the bundle); 2d entries are
filtered from the run list, matching the reference.

## 2d is pending verification

The `2d` config family is **PENDING VERIFICATION** — do not plan a 2d run on this app today.
Honest wiring status (Phase 28 census, `28-hyperparameter-inventory.csv` rows `2d` / `2d_u`):

- The `2d` config **IS in this bundle's `plans.json`** but has **no model dir on disk**
  (plan-only entry). `2d_u` is not present in this bundle's plans at all (schema-family
  completeness row).
- The app pipeline is **3D-only**: the volume contract is `(C, X, Y, Z)` and the inference
  core asserts 4D tensors.
- Loader evidence from the Phase 28 census: `load_preprocess_params` returns a 2-value-spacing
  params object for `2d` (plans-level data loads fine), while `load_inference_params` raises
  `FileNotFoundError` (no model dir, no folds) — so a 2d run would fail fast, not silently
  misbehave.
- 2d entries are **filtered from the run list**, matching the reference (a 2d entry in
  `plans.json` is skipped when its model dir is absent).

**A real 2d model run is deferred to the separate 2d nnUNet app. Do not point a 2d bundle at
this app.**

## Adapting to your own nnUNet bundle

This app is the **template** other `nnunet-fast` apps are ported from: the guide below
teaches the general mechanics; facts that are specific to the shipped airway app (the
`{0: background, 1: airway}` label table, the 15-checkpoint bundle) are called out as such.
A [Config support matrix](#config-support-matrix) per layout × fold count follows below.

**Get your bundle.** Two paths:

1. **The in-repo reference airway bundle** (copy-from-reference-app path — the exact path
   used in the Phase 28–30 evidence): copy the reference app's `models/` tree into this
   app's (git-ignored) `models/` dir and make it world-readable:

   ```bash
   cp -a examples/apps/cchmc_nnunet_fifteen_ckpt_app/models examples/apps/cchmc-nnunet-fast/models
   chmod -R a+rX examples/apps/cchmc-nnunet-fast/models
   ```

   The tree is 1.8 GB: three 3d configs × `fold_0..fold_4` (15 real per-fold
   `best_model.pt`) + `jsonpkls/`. (cchmc-specific: single-foreground-label airway MR —
   `dataset.json` labels are `{airway: 1, background: 0}`; see fixed row 7 below.)
2. **Any other nnUNet bundle.** It must satisfy the layout contract below
   (`jsonpkls/plans.json` + `dataset.json` and per-config `fold_N` dirs). How you train or
   obtain a bundle is out of scope here (per the nnU-Net docs); what the app *requires* is
   the contract.

**The layout contract.** The root resolver (`find_jsonpkls_dir`) accepts a bundle root as
either `<root>/jsonpkls/plans.json` or `<root>/models/jsonpkls/plans.json` (fail-fast: the
error names both tried paths). Under the root:

```
jsonpkls/
├── plans.json              (required)
├── dataset.json            (required)
└── postprocessing.pkl      (optional — the reference bundle ships one with zero rules;
                            the app's generic rule pass is a correct no-op for it)
<config_name>/              # e.g. 3d_fullres / 3d_lowres / 3d_cascade_fullres
├── nnunet_checkpoint.pth   (optional metadata sidecar)
└── fold_N/
    └── final_model.pt      (preferred; best_model.pt / model.pt also accepted)
```

- **Checkpoint preference:** `final_model.pt` → `best_model.pt` → `model.pt`. The reference
  bundle uses `best_model.pt` (no `final_model.pt` on disk) — the auto-order falls through,
  exactly as reference nnU-Net would.
- **Plan-only configs:** configs listed in `plans.json` with no model dir on disk (like this
  bundle's `2d`) are **silently filtered from the run list** at startup — reference
  semantics, by design (see [2d is pending verification](#2d-is-pending-verification)).
- **`HOLOSCAN_MODEL_LIST`:** selects an explicit comma-separated config subset. Unset =
  reference default semantics: run all on-disk configs in `plans.json` order with `3d_lowres`
  excluded from the ensemble, and a cascade config **auto-inserts its `previous_stage`
  producer** — a documented fast-app extension (the reference nnU-Net CLI *crashes* on a
  cascade config without its previous-stage probability maps on disk).

**Run it.** Dev-run command shape — the exact shape the Phase 28/29 gate runs executed
(20/20 cells PASS), consistent with the [Running](#running) section above:

```bash
source activate-env.sh          # activates the project uv venv
cd examples/apps/cchmc-nnunet-fast
ulimit -s unlimited             # Holoscan 4.x wants a large thread stack

/tmp/<venv>/bin/python my_app \
    -i <input_dicom_folder> \
    -m <models_dir> \            # a bundle root per the layout contract above
    -o <output_folder>
```

Env knobs a bundle adaptor may need (full table in [Running](#running)):
`HOLOSCAN_MODEL_LIST` (explicit config subset), `CUDA_VISIBLE_DEVICES` (GPU pin — all gate
runs used GPU 0), `HOLOSCAN_CONCURRENT_FRAGMENTS`, `HOLOSCAN_GPU_RESAMPLE`,
`HOLOSCAN_KEEP_LOWRES_WEIGHTS`.

**What gets produced.** `SEG/` (DICOM-SEG) + `SR/` (measurements) + `SC/` (contour
overlay). **No numpy output** — cchmc has no `.npy` writer (do not import the
totalsegmentator-fast `HOLOSCAN_EMIT_NPY` language).

> **Source:** command shape + contracts reflect the Phase 28/29 gate runs — 20/20 cells
> PASS (`.planning/phases/29-cchmc-gate-verification/29-gate-runs.csv`; runner evidence in
> the phase dir). Bundle facts (folds, `best_model.pt`, zero-rule pkl, plan-only `2d`)
> from `.planning/phases/28-cchmc-nnunet-fast-config-handling-hyperparameter-audit/28-hyperparameter-inventory.md`.

## Config support matrix

Status per nnUNet config layout × fold count. Status vocabulary (strict): **verified** =
explicitly exercised and asserted in a committed Phase 29 gate cell (20/20 cells,
exit 0 + contract PASS in `.planning/phases/29-cchmc-gate-verification/29-gate-runs.csv`;
aggregates in `29-gate-report.json`); **inferred-only** = code-generic / honored per the
Phase 28 inventory but NOT varied in Phase 29 (each row states what would verify it);
**pending** = not runnable today.

> **cchmc honesty note:** unlike the totalsegmentator-fast v3.3 matrix, every 3d cell below ran
> on the **REAL** airway bundle — genuinely distinct per-fold weights (5/5 fold-pair argmax
> diffs non-zero, 0.194–0.335; sha256-pairwise-distinct fold weights per config in
> `29-bundle-manifest.csv`) and real per-config weights. So "verified" here is
> **real-ensemble verification, NOT weights-reused machinery proof** (that caveat belonged to
> the v3.3 ts-fast matrix, whose registry shipped fold_0 only).

| Layout (`HOLOSCAN_MODEL_LIST`) | 1-fold | 5-fold | Notes + evidence |
|---|---|---|---|
| `3d_fullres` (fullres_only) | **verified** | **verified** | f1: 26 s, 1 load, fg 2487; f5: 46 s, 5 loads; f5-vs-f1 argmax diff 0.2156. Rows `fullres_only_f{1,5}_tta_{on,off}` in `29-gate-runs.csv`; pair in `29-gate-report.json` |
| `3d_lowres` (lowres_only) | **verified** | **verified** | f1: 26 s, fg 2299; f5: 45 s; diff 0.3352 (largest). Standalone lowres + documented self-ensemble fallback. `lowres_only_*` rows |
| `3d_cascade_fullres` (cascade_only; auto-expands to `3d_lowres` + cascade) | **verified** | **verified** | f1: 36 s, 2 loads; f5: 76 s, 10 loads; diff 0.2337. Real lowres producer weights, zero-disk-I/O cascade edge. `cascade_only_*` rows |
| `3d_fullres` + `3d_lowres` (fullres_lowres, 2-config bundle) | **verified** | **verified** | f1: 32 s, fg 2487; f5: 72 s; diff 0.2156; ensemble = `[3d_fullres]` via lowres exclusion (config:279). `fullres_lowres_*` rows |
| full bundle = default, unset (full_bundle; v1.0 default path) | **verified** | **verified** | f1: 42 s; f5: 103 s, 15 loads; diff 0.1944; run `[3d_fullres, 3d_lowres, 3d_cascade_fullres]`, ensemble `[3d_fullres, 3d_cascade_fullres]` — the PERF-03 parity path (99.970931% vs v1.0 GT, `29-parity-report.json`). `full_bundle_*` rows |
| `2d` / `2d_u` | **pending** | **pending** | Plan-only entry / absent from plans; 2d is pending — see [2d is pending verification](#2d-is-pending-verification) (no model dir on disk; 3D-only pipeline) |

All cited rows are in `.planning/phases/29-cchmc-gate-verification/29-gate-runs.csv` and `29-gate-report.json`.

### Key hyperparameters

| Setting | Status | Evidence / what would verify it |
|---|---|---|
| `use_mirroring` (TTA on/off) | **verified** | Both directions A/B'd: 10/10 on/off pairs non-zero, inside the user-calibrated airway bar 0 < frac < 0.30 (Decision A, 2026-09-08; measured 0.0954–0.2246 — `29-gate-report.json` `hyper06_disposition`). Byte-identical same-config rerun controls prove the signal is genuine TTA; TTA-on log line `TTA on: config=3d_fullres mirror_axes=(0, 1, 2) combinations=7`. Plans-driven since 29-01 (`ea704b7`); the bundle's plans set no per-config key, so app default `True` = behavior-neutral vs the reference for this bundle |
| `mirror_axes` (checkpoint-meta `inference_allowed_mirroring_axes`) | **verified** | All three 3d checkpoints carry (0, 1, 2); applied as configured (the `combinations=7` line above); A/B-verified in the same 10/10 pairs (`29-gate-report.json`) |
| fold count / ensemble averaging | **verified** | **Real-ensemble verification:** per-config fold weights sha256-pairwise-distinct (`29-bundle-manifest.csv`); f5 = exactly 5× f1 loads; 5/5 f5-vs-f1 argmax diffs non-zero (0.194–0.335); wall deltas +19..+61 s. **No weights-reused machinery caveat** (that belonged to the v3.3 ts-fast matrix) |
| model list / config selection (`HOLOSCAN_MODEL_LIST`) | **verified** | All five literal layouts exercised as distinct run lists (the 5 rows above), incl. cascade auto-insertion and plan-only-`2d` filtering (`29-gate-runs.csv`) |
| 5-fold wall-time deltas | **verified** | +20 / +19 / +40 / +40 / +61 s per layout (fullres / lowres / cascade / fullres_lowres / full_bundle); `29-gate-report.json` `pair_fold` |
| `spacing_in_mm` | inferred-only | Honored per `28-hyperparameter-inventory.md`; not varied in Phase 29. Verify with: a non-corpus spacing tuple |
| `resampling_fn_*_kwargs.order` / `order_z` (data & seg) | inferred-only | Honored; not varied. Verify with: a different resample order |
| `force_separate_z` | inferred-only | Honored; not varied. Verify with: an anisotropic plan flipping the axis |
| `transpose_forward` / `transpose_backward` | inferred-only | Honored; not varied. Verify with: a non-identity transpose |
| `normalization_schemes` | inferred-only | Honored; not varied. Verify with: a ZScore or multi-channel plan |
| `use_mask_for_norm` | inferred-only | Honored; not varied. Verify with: a plan with `true` |
| `foreground_intensity_properties_per_channel` | inferred-only | Honored; not varied. Verify with: materially different percentiles |
| `patch_size` | inferred-only | Honored; not varied. Verify with: a patch outside [128, 128, 128] |
| network architecture | inferred-only | Honored; not varied. Verify with: another architecture block |
| `previous_stage` (cascade producer) | inferred-only | Honored; exercised inside the verified cascade cells, not varied as a hyperparameter. Verify with: a producer other than `3d_lowres` |
| `postprocessing.pkl` rules | inferred-only | Honored; reference pkl carries zero rules (no-op). Verify with: a pkl with actual rules |
| `labels` / `num_channels` | inferred-only | Honored; this bundle is single-label airway. Verify with: a multi-label dataset |

The 8 fixed + 3 unsupported rows are itemized with their effects in
[Fixed & unsupported settings (what the app will never do)](#fixed--unsupported-settings-what-the-app-will-never-do);
`use_mirroring` was among them at audit time and is now honored (see its row above).

> **Source:** `.planning/phases/29-cchmc-gate-verification/29-gate-runs.csv` (20 per-cell rows),
> `29-gate-report.json` (aggregate + `hyper06_disposition`), `29-bundle-manifest.csv`
> (fold-weight distinctness), `29-parity-report.json` (default-path parity),
> `.planning/phases/28-cchmc-nnunet-fast-config-handling-hyperparameter-audit/28-hyperparameter-inventory.md`
> (honored-row provenance).

## Running

```bash
source activate-env.sh          # activates /tmp/monai-env/.venv (uv)
cd examples/apps/cchmc-nnunet-fast
ulimit -s unlimited             # Holoscan 4.x wants a large thread stack

/tmp/monai-env/.venv/bin/python my_app \
    -i <input_dicom_folder> \
    -m <models_dir> \            # e.g. examples/apps/cchmc_nnunet_fifteen_ckpt_app/models
    -o <output_folder>
```

Output: `SEG/` (DICOM-SEG), `SR/` (measurements), `SC/` (contour overlay).

### Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `HOLOSCAN_MODEL_LIST` | unset = reference bundle | Comma-separated model configs to run (table above) |
| `HOLOSCAN_CONCURRENT_FRAGMENTS` | `1` (ON) | `EventBasedScheduler` (worker_thread_number=5); `0` = serial GreedyScheduler fallback |
| `HOLOSCAN_GPU_RESAMPLE` | `1` (ON) | GPU resampling via stock `cupyx.scipy.ndimage` at the three resample call sites; `0` = scipy CPU path (byte-for-byte Phase 2 behavior) |
| `HOLOSCAN_KEEP_LOWRES_WEIGHTS` | unset (release) | `1` = keep `3d_lowres` weights after the aux fragment (opt out of MEM-003) |
| `CUDA_VISIBLE_DEVICES` | — | Pin the GPU (tenancy on shared boxes fluctuates; all benchmarks pin device 0) |

## Performance (A100-SXM4-40GB, single dev study, fresh process, warmup excluded)

| Configuration | Reference app | Phase 1 | Phase 2 | **Phase 3 (shipped)** |
| --- | ---: | ---: | ---: | ---: |
| Bundle (all 3 models) | 169.7 s | — | 129.5 s | **104.2 s (1.629×)** |
| Single fullres | — | 61.8 s | 57.1 s | **49.7 s (1.244× vs Ph 1)** |

Data: `.planning/benchmarks/{baseline_results,baseline-2026-08-18,phase2_results,phase3_results}.csv`;
reports: `.planning/phases/02-gpu-acceleration/02-BENCHMARK-REPORT.md`,
`.planning/phases/03-optimization/03-BENCHMARK-REPORT.md`.

## Correctness & gates

The correctness anchor is the 4-config pixel-exact gate suite against fresh
reference-app oracles (byte-identity + IoU + SR volume + GPU residency):

```bash
# individual pieces (from the app dir unless noted):
/tmp/monai-env/.venv/bin/python scripts/pixel_diff.py <fast_out> <oracle_out>
/tmp/monai-env/.venv/bin/python scripts/gpu_residency.py --static     # or --runtime
# full suites (from repo root):
/tmp/monai-env/.venv/bin/python .planning/scripts/phase2_gate.py --report <out.json>
/tmp/monai-env/.venv/bin/python .planning/scripts/phase3_benchmark.py
```

Headless unit suites: `scripts/test_gpu_bootstrap.py` (RMM import order),
`scripts/test_mem_budget.py`, `scripts/test_cascade_config.py`,
`scripts/test_buffer_cache.py`, `scripts/test_weight_release.py`,
`scripts/test_gpu_zoom_verdict.py`.

Known correctness boundaries (documented, gated, stable):
- fullres-only gate: 99.99986% SEG byte-identity — 3 voxels at the
  documented FP16↔FP32 TTA-accumulation boundary (reference accumulates
  TTA in FP16; this app in FP32).
- Residency: exactly one deliberate `.cpu()` at the postprocess → writer
  boundary (reason-string allow-list in `scripts/gpu_residency.py`).

## Fixed & unsupported settings (what the app will never do)

1:1 with
`.planning/phases/28-cchmc-nnunet-fast-config-handling-hyperparameter-audit/28-hyperparameter-inventory.csv`
(29 rows total at Phase 28: 17 honored / 9 fixed / 3 unsupported; `use_mirroring` since
honored — see the last item below; Phase-29 dispositions in `29-GATE-REPORT.md`).

**Fixed (8) — the app never reads these from the plans; each is benign for this bundle:**

1. **`resampling_fn_probabilities_kwargs.order` / `order_z`** — the plans key is never read;
   the app uses module constants `PROBABILITY_RESAMPLE_ORDER=1` / `_Z=0` on the
   logits→original-space resample. Benign: this bundle's census values equal the constants
   on all 4 configs. Would diverge for a bundle whose plans set different prob-resample
   orders.
2. **`preprocessor_name`** — key never read; the app unconditionally replicates the
   `DefaultPreprocessor` chain. Benign: every raw config in this bundle carries
   `DefaultPreprocessor`.
3. **`border_mode` / `border_value`** — no 2.8.1 plans key exists; the app hardcodes resize
   `mode='edge'` + pad `0` — the reference's own hardcodes. Never honors an alternate mode.
4. **crop-to-nonzero (inference cropping)** — always crops to the nonzero-voxel bbox, exactly
   like reference inference. The training-only `mask_based_cropping` plan key is never
   consulted.
5. **`step_size` (tile step size)** — not plan-configurable in 2.8.1 (a CLI arg there); the
   app pins `DEFAULT_TILE_STEP_SIZE = 0.5` everywhere = the reference default for this
   bundle. Never read from a plan.
6. **`use_gaussian` (+ blend parameters `sigma_scale` / `value_scaling_factor`)** —
   hardcoded `True` at all wiring sites = the reference default; the blend calls the
   reference's own `compute_gaussian` with stock defaults (1/8, 10) — divergence on the
   gaussian parameters is structurally impossible. Never runs without the gaussian blend.
7. **`output_labels` / label names** (app-constant, not a plans key) — `OUTPUT_LABELS = [1]`
   + `{0: background, 1: airway}` are hard-coded (cchmc-specific: matches this airway bundle
   exactly). A bundle with different labels requires an app change — a known airway-app
   limitation; the app does NOT derive labels from the checkpoint's label manager.
8. **memory budget / ensemble defer** — runtime-VRAM behavior, not a plans setting;
   `safety_factor = 1.15` is fixed. Benign: the defer branch is a bit-identical running
   mean (mathematically the reference `average_probabilities` sum/n) — only peak VRAM is
   affected, never results.

**Unsupported (3) — the app cannot run them today:**

9. **`allow_torch_anisotropy`** — absent from this bundle's plans AND 0 consumers in vendored
   nnUNet 2.8.1; nothing to honor today. This row flips if a future plans format
   introduces a consumer.
10. **`2d` (config family)** — in `plans.json` but no model dir on disk; 3D-only pipeline
    (volume contract `(C,X,Y,Z)`; inference core asserts 4D); `load_preprocess_params`
    returns a 2-value-spacing object while `load_inference_params` raises
    `FileNotFoundError`; 2d entries are filtered from the run list. See
    [2d is pending verification](#2d-is-pending-verification).
11. **`2d_u` (config family)** — not even present in this bundle's plans (schema-family
    completeness row); same 3D-only situation as `2d`. Never runs 2d_u.

**`use_mirroring` — RESOLVED (was fixed in the Phase 28 inventory).** The Phase 28 audit
classified it fixed (cchmc read no per-config plans key; constructor default `True`).
Phase 29 (29-01, `ea704b7`) made it **plan-driven via `resolve_use_mirroring`**
(per-config key > top-level key > default `True`) and A/B-verified it 10/10 under the
corpus-calibrated 0.30 airway bar (measured 0.0954–0.2246; user Decision A, 2026-09-08 —
`29-GATE-REPORT.md` §2). It is no longer in the fixed set. Because this bundle's `plans.json`
sets no per-config key, the app default (`True`) applies = behavior-neutral vs the
reference for this bundle.

## Layout

```
my_app/
  app.py                     # DAG composition, multi-fragment factory, scheduler
  gpu_bootstrap.py           # RMM-first bootstrap (MUST import before holoscan)
  mem_budget.py              # memory budget calculator (BudgetPlan)
  operators/
    preprocess_operator.py   # GPU preprocess (CuPy + GPU resample flag)
    slidewindow_operator.py  # one-shot model load, TTA, shape cache, release()
    postresample_operator.py # GPU resample, lowres_seg emit, release_fn hook
    ensemble_average_operator.py
    postprocess_operator.py  # CuPy connected components, exactly-once D2H
    buffer_cache.py          # INFR-02 shape-keyed GPU buffer caches
    gpu_zoom.py              # (provenance) discarded custom RawKernel zoom, D-22a
    gpu_util.py              # app-keyed timing/NVTX helpers
  config/__init__.py         # resolve_run_model_list (reference semantics)
scripts/                     # gates + headless test suites (above)
```

## Dependencies & environment

- Python >= 3.10 · `monai-deploy-app-sdk` · `nnunetv2` (vendored, editable)
- All testing/running uses the uv venv at `/tmp/monai-env/.venv`
  (`source activate-env.sh`). Do not install `nnunetv2` from PyPI.

| Package | Pin | Notes |
| `holoscan-cu13` | `>=4.0.0,<4.3.0` | Bundles the GXF 4.x runtime (see below) |
| `holoscan-cli` | `>=4.0.0,<4.3.0` | `holoscan` / `monai-deploy` CLIs |
| `cupy-cuda13x` | `>=13.6.0` | `cupy-cuda12x` must NOT coexist |
| `rmm-cu13` | `>=25.10.0` | GPU pool allocator |
| `torch` | cu130 build | CUDA 13 runtime |
| `pydicom` | `>=3.0.0` | SDK code uses the pydicom 3.x API |
| `highdicom` | `>=0.24.0` | 0.22.x is incompatible with pydicom 3.x |

### GXF 4.x runtime libraries (no separate install needed)

The native GXF runtime that `holoscan.flow_graphs` links against
(`libgxf_rmm.so`, `libgxf_std.so`, `libgxf_app.so`, `libgxf_core.so`,
`libgxf_cuda.so`, `libgxf_serialization.so`, `libgxf_ucx.so`, …) is
**bundled inside the `holoscan-cu13` wheel** under `holoscan/lib/`. If you
see `ImportError: libgxf_*.so: cannot open shared object file`, the wheel
installation is incomplete — repair:

```bash
uv pip install --python /tmp/monai-env/.venv/bin/python \
    --force-reinstall --no-deps "holoscan-cu13==4.2.0"
```

### GPU driver requirement

A **CUDA 13-capable NVIDIA driver** (R580 series or newer) is required to
run GPU code. With an older driver you will see
`cudaErrorInsufficientDriver`.

### Runtime notes

- Holoscan 4.x wants a large thread stack: `ulimit -s unlimited`
  (or `--ulimit stack=33554432` in Docker).

### Quick environment check

```bash
bash .planning/scripts/validate_venv.sh
/tmp/monai-env/.venv/bin/python -c "import monai.deploy.core, holoscan.flow_graphs; print('stack OK')"
/tmp/monai-env/.venv/bin/python .planning/scripts/test_rmm.py
```

## Building a MAP

The app ships as a Linux x86_64 CUDA-13 MAP container with the model baked in at
build time. Built and verified **in-container** in Phase 30 (2026-09-08/09): image
`ea556134a5f0` (**12.5 GB**) under the established tag
cchmc-nnunet-fast-x64-workstation-dgpu-linux-amd64:latest — **supersedes the v1.1
image `3fee5ae557e7`** (12.4 GB, 2026-08-25, TTA-off era; see the note under
Expected results). The v1.1 build (Phase 5, `.planning/phases/05-batch-validation-map/map_results.md`)
is retained above as history only — the mechanics below are the current ones.

### Build

From the app dir (`examples/apps/cchmc-nnunet-fast/`), with the bundle copy made
world-readable (`chmod -R a+rX <models dir>`):

```bash
monai-deploy package my_app -c my_app/app.yaml \
    -m <abs path to models dir> \
    --monai-deploy-sdk-file <local SDK wheel, clean PEP 440 name> \
    -t cchmc-nnunet-fast:latest \
    --platform x86_64 --cuda 13
# the packager appends the platform suffix ->
#   cchmc-nnunet-fast-x64-workstation-dgpu-linux-amd64:latest
```

CLI pitfalls (all hit the hard way in Phases 24/30):

- **Short `-t` tag** — the packager appends `-x64-workstation-dgpu-linux-amd64`;
  passing the full long tag doubles the suffix.
- **`--monai-deploy-sdk-file`** — the wheel filename must be a clean PEP 440 name;
  a sibling wheel with extra markers (the `pinhsc420` build) is rejected by the
  in-image pip.
- **`-c my_app/app.yaml`** — the CLI resolves `-c` against the app dir; the bare
  `app.yaml` form does not work.

**Runtime nnunetv2 comes from the PyPI pin `nnunetv2==2.8.1` in
`my_app/requirements.txt` — there is NO `--add` bake of any kind.** The v1.1
`--add <trimmed vendored nnunetv2>` pattern is **retired** (that is the main thing
this rewrite replaces in the old v1.1-era text). What gets baked: the `-m` dir's
**contents** land at `/opt/holoscan/models/` — for the reference airway bundle that is
`/opt/holoscan/models/{3d_fullres,3d_lowres,3d_cascade_fullres,jsonpkls}/` (15
`best_model.pt` + jsonpkls verified in-image).

### Run

```bash
monai-deploy run cchmc-nnunet-fast-x64-workstation-dgpu-linux-amd64:latest \
    -i <input-dicom-dir> \
    -o <output-dir> \
    --gpus <N> \
    --uid 1000 --gid 10000
```

`--uid 1000 --gid 10000` are **required**: the in-image `holoscan` user (the app's
packages live in its user-site) plus the `/data` group — the CLI default host uid
fails at import and on permissions. The host output dir must be writable by
uid 1000 (`chmod 0777` it). Input: a single-study DICOM series directory (one
`.dcm` per slice); the app's series-selector rule picks the thin-slice MR
series.

### Caveats

- **Exit status:** `monai-deploy run` returns 0 **even on app failure** (holoscan-cli
  4.2) — the true signal is the app-log line `Application exited with N`.

### Expected results

- App log `Application exited with 0` and non-empty `SEG/` (DICOM-SEG) + `SR/`
  (measurements) + `SC/` (contour overlay). **cchmc emits no numpy `.npy`** (no npy
  writer in the app).
- Verified on the 256³ airway study (Phase 30): wall **≈ 117 s** incl. container
  start + model load (A100); in-container SEG fg = **2,447 voxels** = the
  standalone / Phase 29 default-path foreground — the baked 15-fold bundle + v3.4
  code reproduce the reference segmentation in-container. Container overhead
  **+7 s** vs the same-study pythonic standalone wall (110 s) = **IN-CLASS** vs the
  v1.1 +5–8 s container-overhead class. Note the old v1.1-era "68–71 s (256³)" / "142
  s (384³)" numbers were **TTA-off** and are superseded — plan-driven `use_mirroring`
  (default ON) is the new baseline (see [Fixed & unsupported settings](#fixed--unsupported-settings-what-the-app-will-never-do)).

### In-image verification (one-liners)

```bash
docker run --rm --entrypoint python3 cchmc-nnunet-fast-x64-workstation-dgpu-linux-amd64:latest \
    -c "from importlib.metadata import version; print(version('nnunetv2'))"   # -> 2.8.1
docker run --rm --entrypoint bash cchmc-nnunet-fast-x64-workstation-dgpu-linux-amd64:latest \
    -c "grep -c resolve_use_mirroring /opt/holoscan/app/config/__init__.py"   # -> 3 (v3.4 code baked;
                                      # a v1.1-era image would fail this)
```

> **Source:** build/run commands and all numbers verbatim from Phase 30 evidence —
> `.planning/phases/30-map-build-in-container-verification/30-cchmc-build-log.md` (build
> + in-image proofs), `30-map-verification.md` (in-container run + 35/35 contract
> asserts), `30-parity-timings.md` (117 s vs 110 s, +7 s IN-CLASS, TTA-era note).

## Known limitations / external dependencies

- **≥5-CT corpus gate**: the pixel-exact suite currently runs on a single
  dev study; the multi-CT re-run is blocked on CT data.
- **2d config**: blocked on a real 2d model (wiring is generic).
- **Inference-kernel tuning / TensorRT**: blocked on `ncu` admin access
  (`ERR_NVGPUCTRPERM`); nsys-only profiling is in place.
- **8 GB VRAM target** (MEM-02): unverifiable on the 40 GB dev GPU;
  incremental (defer) strategies are implemented and unit-tested.

## History

See `.planning/MILESTONES.md` (v1.0 entry), `.planning/RETROSPECTIVE.md`,
and the per-phase artifacts under `.planning/phases/`.
