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

Docker packaging + container test is the first deferred item (v2.0). The
conventional entry point remains:

```bash
monai-deploy package my_app -m <model_path> -c my_app/app.yaml -t <tag>
```

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
