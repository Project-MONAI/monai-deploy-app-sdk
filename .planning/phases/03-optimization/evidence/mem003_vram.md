# MEM-003 (D-23) — Peak-VRAM delta evidence: 3d_lowres weight release

**Device:** GPU 0, NVIDIA A100-SXM4-40GB (40,960 MiB) — pinned per Pitfall 7 (devices 0–3 free at run time; 4–7 tenant-occupied).
**Date:** 2026-08-19 (UTC)
**Shipping state under test:** concurrent fragments default ON (D-21, `HOLOSCAN_CONCURRENT_FRAGMENTS` unset → EventBasedScheduler, confirmed in the run log: `scheduler: EventBasedScheduler worker_thread_number=5 (D-21)`), RMM `initial_pool_size` pinned 4 GiB (Plan 01), `HOLOSCAN_MODEL_LIST` **unset** (3-config bundle / cascade path).
**Measurement:** pynvml `nvmlDeviceGetMemoryInfo` at 2 Hz (`.planning/scripts/vram_sampler.py`) around one full bundle rep. torch's CUDA memory counters are NOT used anywhere (they raise under the RMM pluggable allocator — RESEARCH Pitfall 2; verified by grep across this plan's files).

**Sampler CSVs (committed, epoch-ns timestamps):**
- release ON: `evidence/mem003_vram_on.csv` (210 rows, run log `evidence/mem003_run_on.log`)
- release OFF (env opt-out `HOLOSCAN_KEEP_LOWRES_WEIGHTS=1`): `evidence/mem003_vram_off.csv` (211 rows, run log `evidence/mem003_run_off.log`)

Both runs: app exit 0, exactly one `study_timing_summary` per run. The ON run's log contains the release line:

```
[2026-08-19 11:35:15,282] [INFO] (my_app.operators.slidewindow_operator.SlideWindowOperator) - weights released: 3d_lowres (folds=5) (MEM-003)
```

## (a) Driver-level used VRAM at 3 moments (release ON, device 0)

| Moment | Log timestamp (local) | Nearest 2 Hz sample | Driver used VRAM |
|---|---|---|---|
| pre-lowres-inference (`inference_3d_lowres` start) | 11:34:22.436 | 11:34:22.477 | 5.413 GiB |
| at the `weights released: 3d_lowres` line (±1 s window) | 11:35:15.282 | 11:35:15.045 | 9.427 GiB |
| post-cascade-complete (`postresample_3d_cascade_fullres` end) | 11:35:51.637 | 11:35:51.589 | 9.552 GiB |

The 5.4 → 9.4 GiB rise is the concurrent fullres∥lowres inference phase (D-21) building up the RMM pool; the release itself is invisible at this level (no drop after 11:35:15.282 — the row sequence stays flat at ~9.4–9.55 GiB through cascade inference and post-cascade-complete).

## (b) Peak driver VRAM — release ON vs OFF

| Run | Peak driver used | Peak timestamp (local) |
|---|---|---|
| release ON (shipping) | **9.552 GiB** | 11:35:24.557 (during cascade inference) |
| release OFF (`HOLOSCAN_KEEP_LOWRES_WEIGHTS=1`) | **9.552 GiB** | 11:37:45.271 (during cascade inference) |

**Delta: 0.000 GiB — the driver-level peak is identical.**

## (c) Pool level vs driver level — which level moved

- **Pool level: down ~0.8 GiB after the release — DERIVED, not directly measured.** rmm 26.2.0's Python API exposes no pool statistics (probed: `[a for a in dir(rmm) if 'pool' in a.lower() or 'size' in a.lower()]` → `[]`), so pool occupancy cannot be sampled directly. The delta is derived from the checkpoint arithmetic (3d_lowres = 5 × 135 MB fold state dicts + ~135 MB live network ≈ 0.8 GB, RESEARCH §MEM-003): `release()` deterministically drops the only Python references to those CUDA tensors (`_bundle = None`, `del network/fold_state_dicts`) and the RMM pool's free list reclaims the blocks — no downstream code path touches the released bundle (the cascade consumes only the emitted `lowres_seg` tensor; unit-tested, `scripts/test_weight_release.py`).
- **Driver level: FLAT — directly measured.** Both the 2 Hz sampler (0.000 GiB peak delta, no drop after the release line) and the headless unit test (case b: pynvml 5,019 → 5,019 MiB, `+0 MiB`, after releasing the real 0.8 GB 3d_lowres bundle under the 4 GiB RMM pool) show zero driver-level movement.

**Conclusion (honest, per-level):**

1. **Pool level: the ~0.8 GB IS freed** (derived: reference drop + pool reclamation; the rmm pool returns freed blocks to its internal free list, which is what torch-side allocations draw from).
2. **Driver level: flat is the valid, expected result — Open Q2 answered: `torch.cuda.empty_cache()` does NOT return RMM pool memory to the driver on rmm 26.2.0** (it never errors either — a silent driver-level no-op). The RMM pool never shrinks its cudaMalloc-reserved blocks, so the driver keeps counting them; the peak (9.55 GiB) is set by the concurrent inference phase's pool growth, which precedes the release, and is dominated by the pool reservation anyway (research expectation, confirmed).
3. Practical reading on this 40 GB box: MEM-003 reduces *pool-occupied* (non-reusable-by-other-tenants) VRAM by ~0.8 GB during the cascade phase — meaningful for tighter-memory targets (e.g. the deferred MEM-02 8 GB class) and for headroom before the cascade phase, but it produces **no driver-visible peak-VRAM reduction** on the A100-40GB under the RMM pool, and **no latency change** (this is a memory-lifecycle deliverable, not a speed lever — per D-23 the wall time is unaffected).

## Gate re-run (D-25 anchor, release hook ACTIVE — the shipping state)

`HOLOSCAN_CONCURRENT_FRAGMENTS=1 /tmp/monai-env/.venv/bin/python .planning/scripts/phase2_gate.py --report .planning/phases/03-optimization/gates/03-GATE-mem003.json` — **ALL GATES: PASS**, pixel-identical to Plan 01's concurrent gate: fullres 99.99986% / 3 differing voxels (documented fp16↔fp32 boundary class), lowres 100.00000% / 0, cascade 100.00000% / 0, bundle 100.00000% / 0; SR airway volume 0.0% delta ×4; residency static + runtime PASS; sanity (bundle ensembles: 382 differing voxels vs fullres-only) OK. → `gates/03-GATE-mem003.json` (`all_gates_pass: true`).
