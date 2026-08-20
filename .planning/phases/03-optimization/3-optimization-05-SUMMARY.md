# Phase 3 Plan 05 Summary — Close-Out: 2×2 Benchmark + Final Gate Suite + Benchmark Report

**Status: COMPLETE — Phase 3 5/5 plans shipped (pending /gsd-verify-phase).**

The final plan measured the whole phase: the 2×2 matrix
(`HOLOSCAN_CONCURRENT_FRAGMENTS` × `HOLOSCAN_GPU_RESAMPLE`, both flags set
explicitly per cell) under the Phase 2 methodology (fresh process per rep,
32 MB stack, 1 warmup + 3 measured per cell, warmups excluded, parse-by-
operator-name), the final gate suite in the shipping configuration, and the
two-bar report with the deferred/external-dependency record for
VERIFICATION.md.

## Deliverables

| Artifact | Content |
|---|---|
| `.planning/scripts/phase3_benchmark.py` | Matrix harness (copy-extended from `phase2_benchmark.py` per D-25 — not rebuilt): `--scopes`/`--cells`/`--gpu`/`--timeout-s` CLI, explicit env per cell, `cell` + `gpu` CSV columns (Pitfall 7), per-rep 900 s `subprocess` timeout, per-cell mean±std summary rows, append-safe CSV |
| `.planning/benchmarks/phase3_results.csv` | 40 rows: 2 scopes × 4 cells × (1 warmup + 3 measured) + 8 summary rows; GPU 0 in every row; all 32 reps `ok=true` |
| `.planning/phases/03-optimization/03-GATE-RESULTS.json` | Final shipping-configuration gate — `all_gates_pass: true` + a top-level `note` recording the four shipped flag defaults + device pin |
| `.planning/phases/03-optimization/03-BENCHMARK-REPORT.md` | The 8-section two-bar report (§1 scope/method, §2 same-scope bar vs 61.8 s AND 57.14 s, §3 headline bundle bar vs 169,747 ms + scope-asymmetry note, §4 full 2×2 with isolated factor effects, §5 optimization rollup vs the Phase 2 §6 ranked items, §6 deviations & honesty, §7 deferred-with-reason, §8 D-26 external dependencies — §7/§8 verbatim-liftable for VERIFICATION.md) |

## Two-bar numbers (D-18) — shipping configuration = `conc-resample-on` cell

**Same-scope fullres (`HOLOSCAN_MODEL_LIST=3d_fullres`): 49,672.9 ± 166.7 ms**
- **1.244× vs Phase 1's 61.8 s** (`baseline-2026-08-18.csv`)
- **1.150× vs Phase 2's 57,140.3 ± 250.4 ms** (`phase2_results.csv`) — Δ −7,467 ms (−13.1%); the increment is the GPU resample: `preprocess_ms_3d_fullres` 7,628.8 → 2,279.3 ms (−70.1%), `postresample_ms_3d_fullres` 1,680.4 → 264.5 ms (−84.3%); inference unchanged within noise (−2.5%)

**Headline bundle (env unset = 3-config reference default): 104,180.1 ± 618.5 ms**
- **1.629× vs the 169,747 ± 7,274 ms reference** (`baseline_results.csv`)
- **1.243× vs Phase 2's 129,542.9 ± 896.4 ms** — Δ −25,363 ms (−19.6%)
- Bundle-vs-61.8 s prints as 0.593× WITH the D-18 scope-asymmetry note (never a bar)

## The 2×2 matrix headline (E2E means, n=3, warmups excluded)

| | resample OFF (scipy) | resample ON (CuPy, shipping) |
|---|---:|---:|
| **fullres / concurrent (shipping)** | 56,459.8 | **49,672.9** |
| fullres / serial | 56,649.6 | 49,288.6 |
| **bundle / concurrent (shipping)** | 115,141.3 | **104,180.1** |
| bundle / serial | 127,172.7 | 107,021.1 |

- **No NA cells** — Plan 04 shipped the resample flag ON, so the ON column ran;
  OFF = `HOLOSCAN_GPU_RESAMPLE=0`.
- **No regressed cell**; the `serial-resample-off` cell reproduces Phase 2
  within −0.85% (fullres) / −1.83% (bundle) — methodology and environment stable.
- **Isolated effects:** GPU resample = the dominant lever (−6.8/−7.4 s fullres;
  −11.0/−20.2 s bundle at conc/serial; resample-dominated bundle spans
  28.8 s → 9.65 s — Phase 2's ranked #1 bottleneck, exactly as predicted).
  Concurrency = second lever (−12.0 s / −9.5% on the Phase 2 configuration;
  −2.8 s on the shipping configuration, because the concurrent scheduler had
  already partially overlapped the CPU-bound scipy spans with GPU inference in
  Phase 2). Per-fragment inference walls double under concurrency (25 s → ~50 s,
  total GPU work conserved — the Plan-01 ceiling, measured, not hidden).

## Final gate (shipping configuration → 03-GATE-RESULTS.json)

`CUDA_VISIBLE_DEVICES=0 /tmp/monai-env/.venv/bin/python .planning/scripts/phase2_gate.py --report .../03-GATE-RESULTS.json`, all four plan flags at their shipped defaults (concurrent ON, resample ON, caches active, release hook active; RMM 4 GiB pin) — **ALL GATES PASS, pixel-identical to the Phase 2/3 baseline:**

| row | byte-identity / differing voxels | SR delta |
|---|---|---:|
| fullres_only | 99.99986% / 3 (documented fp16↔fp32 boundary class, IoU 0.998714) | 0.0000% |
| lowres_only | 100.00000% / 0 | 0.0000% |
| cascade_only | 100.00000% / 0 | 0.0000% |
| bundle | 100.00000% / 0 (2447=2447 vs testdata/current_output) | 0.0000% |

Residency static + runtime PASS; sanity (bundle actually ensembles: 382
differing voxels vs fullres-only) OK. Every row matches the known Plan 01–04
baseline — the Phase 3 optimizations moved the wall clock without moving a
voxel. This run stands as the TEST-02 (SR 0.1% bar) + TEST-003 (automated
pixel-diff fails on divergence) deliverable for the phase.

## Deviations

- `write_ms` runs ~+554 ms higher in this session's reps than Phase 2's
  (DICOM I/O rep-level variance) — flagged in the report (§2 note ¹), no
  verdict impact.
- The plan's resample-ON-cell "NA rows" clause did not trigger (Plan 04
  shipped the flag ON under the amended D-22b gate).
- No other deviations — the plan ran as specified (8 cells ran; the plan's
  expected 45–55 min wall measured ~48 min).

## Commits

1. `a04fa48` perf(03-05): 2x2 benchmark matrix measured (resample x concurrent, 8 cells, gpu=0) -> phase3_results.csv
2. `72bc067` perf(03-05): final shipping-config gate green (pixel-identical to baseline) + two-bar 2x2 benchmark report

## Handoff for /gsd-verify-phase (VERIFICATION.md)

- **What to check:** (1) the two bars as stated in `03-BENCHMARK-REPORT.md`
  §2/§3 against the cited CSVs (`phase3_results.csv`, `phase2_results.csv`,
  `baseline-2026-08-18.csv`, `baseline_results.csv`); (2) `03-GATE-RESULTS.json`
  `all_gates_pass: true` + the pixel-identity numbers above; (3) the
  ROADMAP Phase 3 acceptance criteria (all five now checked, each with its
  evidence path); (4) the per-v2-requirement decision record.
- **§7 and §8 of the report are verbatim-liftable** — the deferred-with-reason
  list (ACCEL-01/02/03 ncu-admin-blocked; MEM-01 not a measured bottleneck;
  MEM-02 hardware-unverifiable on A100-40GB; pylibraft 3.5 evaluated-not-taken)
  and the three D-26 external dependencies (≥5-CT corpus re-run = TEST-01
  final; ncu admin access; INFR-02 user reference examples) are self-contained;
  each external dependency states it re-opens as a gap plan if the dependency
  lands.
- **Requirement status:** TEST-01/TEST-002/TEST-003/TEST-006/TEST-007 met on
  the dev corpus with the §7/§8 deviations; TEST-01's ≥5-CT half is external
  dependency (1).
