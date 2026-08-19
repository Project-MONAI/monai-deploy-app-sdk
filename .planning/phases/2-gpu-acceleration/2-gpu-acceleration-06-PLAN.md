---
phase: "2-gpu-acceleration"
plan: "06"
type: "execute"
wave: 5
depends_on: ["05"]
files_modified:
  - ".planning/scripts/phase2_benchmark.py"
  - ".planning/benchmarks/phase2_results.csv"
  - ".planning/profiles/phase2/"
  - ".planning/phases/2-gpu-acceleration/02-BENCHMARK-REPORT.md"
autonomous: true
requirements: [TEST-006, TEST-007, INFR-005]
must_haves:
  truths:
    - "A benchmark script measures E2E + per-operator (per-config) latency for the fast app with fresh-process reps, 1 warmup + >=3 measured reps, mean +/- std (TEST-006)"
    - "BOTH latency bars are reported (resolved D-18 tension): (a) SAME-SCOPE fullres-only vs Phase 1's 61.8 s — the 'positive improvement' bar — and (b) HEADLINE bundle vs the 169.7 s reference baseline, with per-operator deltas vs Phase 1 and vs 169.7 s (TEST-007, D-18)"
    - "The bundle-vs-61.8 s comparison is reported WITH the scope difference documented as a deviation from the literal D-18 wording (Phase 1's 61.8 s was fullres-only; a serial 3-config bundle is estimated 70-80 s by RESEARCH.md) — the report states the bar explicitly, not silently"
    - "Results saved to .planning/benchmarks/phase2_results.csv (roadmap acceptance criterion) with per-config columns parseable from the app's `timing: {...}` JSON logs"
    - "nsys traces + stats exports saved to .planning/profiles/phase2/; `cuda_api_sum` shows cudaMalloc/cudaFree only in the setup/warmup span (RMM active, no per-tile churn); per-config NVTX ranges legible (INFR-005, task 2.14)"
    - "ncu is documented as UNAVAILABLE (ERR_NVGPUCTRPERM permission block, verified 2026-08-19) — no fake kernel-level metrics; the admin requirement is noted (task 2.13)"
    - "Stream-overlap observation is recorded honestly (visible overlap = plus; absence = note, D-16)"
  artifacts:
    - path: ".planning/scripts/phase2_benchmark.py"
      provides: "fast-app E2E + per-config benchmark harness (subprocess, timing-log parsing, CSV writer)"
      contains: "phase2_results"
    - path: ".planning/benchmarks/phase2_results.csv"
      provides: "the roadmap-required benchmark CSV (scope, rep, per-operator ms, speedups)"
    - path: ".planning/profiles/phase2/"
      provides: "nsys .nsys-rep + .sqlite + stats exports (cuda_api_sum, nvtx ranges, kernel timeline) for the bundle run"
    - path: ".planning/phases/2-gpu-acceleration/02-BENCHMARK-REPORT.md"
      provides: "the two-bar report with per-operator deltas, the D-18 deviation, and profiling findings for Phase 3 scoping"
  key_links:
    - from: "fast app `timing: {...}` / `study_timing_summary:` JSON logs"
      to: "phase2_results.csv per-config columns"
      via: "regex parse of the structured logs (Plan 04 made them config-tagged — INFR-005/Pitfall 9)"
      pattern: "timing: "
    - from: "nsys trace"
      to: "cudaMalloc churn check"
      via: "nsys stats --report cuda_api_sum; cudaMalloc/cudaFree only in the setup/warmup span (RMM pool)"
      pattern: "cuda_api_sum"
---

# Phase 2 Plan 06: Benchmarks (two-bar, per-operator) + nsys Profiling + Phase 3 Handoff

## Objective
- **What:** Extend the Phase 0 benchmark pattern to the fast app, run the three
  benchmark scopes (fullres-only same-scope; bundle headline; bundle-vs-61.8 s
  documented), profile the bundle run with nsys (ncu is permission-blocked), and write
  the benchmark report with per-operator deltas — the Phase 2 performance acceptance and
  the Phase 3 input.
- **Why:** Roadmap tasks 2.12–2.14 + acceptance criteria ("End-to-end latency improved
  vs. baseline (quantified)", "no per-tile cudaMalloc churn visible", "Benchmark
  comparison report saved to .planning/benchmarks/phase2_results.csv").
- **Output:** `phase2_results.csv`, `.planning/profiles/phase2/` artifacts,
  `02-BENCHMARK-REPORT.md`.

## Execution Environment

- Python: `/tmp/monai-env/.venv/bin/python`; every app run: fresh subprocess from the
  app root with `ulimit -s unlimited` (the `baseline_benchmark.py` pattern — read its
  subprocess/ulimit handling and reuse it).
- nsys: `/usr/local/cuda/bin/nsys` (Nsight Systems 2025.6.3; harness template
  `.planning/scripts/nsight_profile.sh` — `--trace=cuda,nvtx,osrt,cublas,cudnn`; the
  legacy `--trace=cub` flag no longer exists).
- **ncu: BLOCKED** — `ERR_NVGPUCTRPERM` (verified 2026-08-19; profiling counters need
  admin: `NVreg_RestrictProfilingToAdminUsers=0` or sudo). Do NOT run ncu as a gate item;
  document the admin requirement in the report.
- Baselines to compare against:
  - Phase 1: `.planning/benchmarks/baseline-2026-08-18.csv` — 61.2–62.2 s E2E (61.8 s
    center), in-study 42.1 s, inference 27.2 s dominant, per-operator rows.
  - Reference: `.planning/baseline_results.csv` — 169,747 ± 7,274 ms (setup ~12.8 s,
    inference ~138 s, postprocess 9–23 s, write ~1.2 s).
- Commit after each major change.

## Context

@.planning/ROADMAP.md (Phase 2 tasks 2.12–2.14 + Acceptance Criteria)
@.planning/phases/2-gpu-acceleration/02-RESEARCH.md (Pattern 7 "Benchmarking & profiling"; Open Question 1 — the D-18 resolution this plan implements; Pitfall 9)
@.planning/scripts/baseline_benchmark.py (the subprocess + warmups + CSV spine to extend)
@.planning/scripts/nsight_profile.sh (the working nsys harness)
@.planning/benchmarks/baseline-2026-08-18.csv (Phase 1 per-operator numbers)

**The resolved D-18 latency bar (orchestrator + RESEARCH Open Question 1):**
Phase 1's 61.8 s was a SINGLE-CONFIG fullres run; the Phase 2 bundle does ~2x the
inference work (serial estimate 70–80 s). The report therefore carries TWO primary bars:
(a) **same-scope**: fast fullres-only (HOLOSCAN_MODEL_LIST=3d_fullres, the CuPy-ported
  pipeline) vs Phase 1's 61.8 s — this is the "any positive E2E improvement" bar;
(b) **headline**: fast default bundle vs the 169.7 s reference bundle — the
  user-facing speedup (≈2.3–2.5x expected). The bundle number is ALSO printed next to
  61.8 s WITH an explicit scope-difference note (deviation from the literal D-18 wording
  — documented here and in the report, per the orchestrator's resolution).

## Tasks

<task type="auto">
  <name>Task 1: phase2_benchmark.py + the three benchmark scopes → phase2_results.csv</name>
  <files>.planning/scripts/phase2_benchmark.py, .planning/benchmarks/phase2_results.csv</files>
  <read_first>
    - .planning/scripts/baseline_benchmark.py (subprocess launch, sitecustomize logging shim, warmups, CSV writer `study,rep,warmup,total_ms,setup_ms,inference_ms,postprocess_ms,write_ms` — reuse the launch pattern; the fast app needs NO marker parsing beyond its own JSON logs)
    - examples/apps/cchmc-nnunet-fast/my_app/app.py + my_app/operators/gpu_util.py (the exact log formats to parse: `timing: {json}` per operator — fields {operator, study, start, end, duration_ms, config (Plan 04), ...} — and `study_timing_summary: {json}`)
    - .planning/benchmarks/baseline-2026-08-18.csv + .planning/baseline_results.csv (column semantics for the delta tables)
  </read_first>
  <action>
1. Create `.planning/scripts/phase2_benchmark.py`:
   - Args: `--scope {fullres,bundle}` (reps configurable), `--reps 3`, `--warmups 1`,
     `--output-csv .planning/benchmarks/phase2_results.csv` (append-safe: the script
     manages its own header), `--study-dir testdata/airway_input`,
     `--model examples/apps/cchmc_nnunet_fifteen_ckpt_app/models`.
   - Each rep: FRESH subprocess (cold start included — the clinical single-study usage),
     launched from `examples/apps/cchmc-nnunet-fast` with `ulimit -s unlimited` (bash -c
     or preexec_fn setrlimit, per baseline_benchmark.py's working pattern),
     `HOLOSCAN_MODEL_LIST=3d_fullres` for the fullres scope, UNSET for the bundle scope.
     Total = subprocess wall time.
   - Parse the fast app's structured logs: every `timing: {json}` line (operator,
     config, duration_ms) and the final `study_timing_summary: {json}`. Build per-config
     columns: for each config that ran: `preprocess_ms_<cfg>`, `inference_ms_<cfg>`,
     `postresample_ms_<cfg>`; plus `ensemble_ms`, `postprocess_ms`, `write_ms`
     (sum of the writer records), `total_ms`.
   - CSV columns (one row per rep):
     `scope,study,rep,warmup,total_ms,preprocess_ms_<cfg>,inference_ms_<cfg>,postresample_ms_<cfg>,ensemble_ms,postprocess_ms,write_ms,speedup_vs_61.8s,speedup_vs_169.7s`
     (`<cfg>` columns: fullres rows get `..._3d_fullres`; bundle rows get the three
     configs `3d_fullres, 3d_lowres, 3d_cascade_fullres`).
     `speedup_vs_61.8s = 61800/total_ms`, `speedup_vs_169.7s = 169747/total_ms`
     (constants documented in the script header as the Phase 1 center and the reference
     baseline mean).
   - After all reps per scope: print mean ± std (warmups excluded) per column and the
     two speedups, and APPEND a `summary` row (rep="mean±std") to the CSV.
2. Run both scopes (1 warmup + 3 measured reps each; ~1 min/rep for fullres, ~2–3 min/rep
   for the bundle):
   ```bash
   cd /users/srv-mde/projects/monai-deploy-app-sdk
   /tmp/monai-env/.venv/bin/python .planning/scripts/phase2_benchmark.py --scope fullres
   /tmp/monai-env/.venv/bin/python .planning/scripts/phase2_benchmark.py --scope bundle
   ```
3. Verify the CSV is complete and internally consistent (per-rep operator sums within
   ~10% of the in-study portion of total; the fullres scope's `inference_ms_3d_fullres`
   ≈ Phase 1's 27.2 s ± a few s — a large drift indicates an environment change and
   must be noted in the report).
  </action>
  <acceptance_criteria>
    - `grep -n "HOLOSCAN_MODEL_LIST" .planning/scripts/phase2_benchmark.py` returns >= 1 line; `grep -n "timing: \|study_timing_summary" .planning/scripts/phase2_benchmark.py` shows the JSON-log parsing (regex over those log prefixes).
    - `.planning/benchmarks/phase2_results.csv` exists; header contains `scope,study,rep,warmup,total_ms` AND per-config columns for BOTH scopes (`inference_ms_3d_fullres`, `inference_ms_3d_lowres`, `inference_ms_3d_cascade_fullres` in bundle rows) AND `speedup_vs_61.8s,speedup_vs_169.7s`; row count >= 8 (2 scopes x (1 warmup + 3 measured)) + 2 summary rows.
    - The console summary (captured in the report) shows mean ± std over >= 3 measured reps per scope, warmups excluded.
    - Speedups are populated: bundle `speedup_vs_169.7s` expected ~2.0–2.8 (serial estimate); fullres `speedup_vs_61.8s` recorded WHATEVER it is (the positive-improvement bar is evaluated in the report against the measured number).
    - Commits: script + CSV.
  </acceptance_criteria>
  <verify>cd /users/srv-mde/projects/monai-deploy-app-sdk && /tmp/monai-env/.venv/bin/python -c "
import csv
rows=list(csv.DictReader(open('.planning/benchmarks/phase2_results.csv')))
assert len(rows) >= 8, len(rows)
assert all('speedup_vs_61.8s' in r for r in rows)
print('rows:', len(rows), 'OK')
" && head -2 .planning/benchmarks/phase2_results.csv</verify>
  <done>phase2_results.csv carries the roadmap-required per-rep, per-operator, two-speedup benchmark data for both scopes (TEST-006 satisfied for the new app; TEST-007's speedup-ratio half computed against both baselines).</done>
</task>

<task type="auto">
  <name>Task 2: nsys profiling of the bundle run → .planning/profiles/phase2/ (cudaMalloc churn, NVTX, stream-overlap note; ncu documented unavailable)</name>
  <files>.planning/profiles/phase2/, .planning/scripts/nsight_profile.sh (extend or add a phase2 wrapper)</files>
  <read_first>
    - .planning/scripts/nsight_profile.sh (the working nsys 2025.6.3 harness: flag set, capture range)
    - .planning/phases/2-gpu-acceleration/02-RESEARCH.md (Pattern 7 nsys section; Environment Availability table — ncu blocked `ERR_NVGPUCTRPERM`)
    - .planning/profiles/ (Phase 0/1 trace naming convention: `<trace-name>_<timestamp>.nsys-rep`)
  </read_first>
  <action>
1. Extend (or add a sibling wrapper to) the nsys harness for the FAST app bundle
   configuration: nsys profile of
   `HOLOSCAN_MODEL_LIST=` (unset → default bundle) fast-app run on the airway study,
   `--trace=cuda,nvtx,osrt,cublas,cudnn` (keep the Phase 0/1 flag set — NOT the removed
   `cub`), `--capture-range=cudaProfilerApi` if the harness uses it, output to
   `.planning/profiles/phase2/phase2_bundle_<timestamp>.nsys-rep`.
2. Generate and save the stats exports alongside the trace:
   - `nsys stats --report cuda_api_sum <rep>` → `phase2_bundle_cuda_api_sum.txt`
     — CHECK: `cudaMalloc`/`cudaFree` calls occur only in the setup/warmup time span
     (model load + RMM warm pool), not per-tile during study compute. If per-tile churn
     is visible, that is a real finding (RMM not covering those allocations) — report it,
     don't hide it.
   - `nsys stats --report nvtx_kern_sum <rep>` (and/or `nvtx_ranges`) → per-operator
     range visibility: expect ranges `preprocess_3d_fullres`, `inference_3d_fullres`,
     `postresample_3d_fullres`, `preprocess_3d_lowres`, `inference_3d_lowres`,
     `preprocess_3d_cascade_fullres`, `inference_3d_cascade_fullres`,
     `ensemble_average`, `write_seg`/`write_sr`/`write_sc` — legible per-config
     boundaries (INFR-005 acceptance).
   - Kernel timeline export (e.g. `nsys stats --report cuda_gpu_kern_sum <rep>` or the
     GUI-less equivalent available in 2025.6.3) → `phase2_bundle_kernels.txt`, used to
     identify remaining CPU-bound regions (task 2.13) — GPU-idle gaps between NVTX
     ranges are the Phase 3 bottleneck candidates (e.g. the scipy resample spans).
3. Stream-overlap observation (D-16, best-effort): from the kernel timeline / NVTX,
   note whether kernels from different fragments' stream pools
   (`streams_3d_fullres` / `streams_3d_lowres` / `streams_3d_cascade_fullres`) visibly
   overlap. Write the honest observation either way (overlap visible = plus; not visible
   = note with the likely reason, e.g. serial fragment scheduling).
4. ncu: attempt NOTHING. Add to the export set a `ncu_status.txt` containing:
   "ncu (Nsight Compute 2026.1.0) is installed but BLOCKED on this host:
   ERR_NVGPUCTRPERM (verified 2026-08-19 live probe). Kernel-level counter profiling
   requires admin (NVreg_RestrictProfilingToAdminUsers=0 or sudo). nsys
   (cuda_api_sum + NVTX + kernel timeline) is the profiling basis for Phase 2/3."
   (No kernel-metric numbers from ncu may appear anywhere in the artifacts.)
5. Commit the artifacts (or, if the .nsys-rep/.sqlite are too large for the repo
   convention, commit the stats exports + a README listing the trace paths — match how
   Phase 0/1 handled .planning/profiles/ artifacts).
  </action>
  <acceptance_criteria>
    - `ls .planning/profiles/phase2/` shows at least: one `*.nsys-rep` (or README pointing to it per repo convention), `phase2_bundle_cuda_api_sum.txt`, a NVTX report file, a kernel report file, `ncu_status.txt`.
    - In `phase2_bundle_cuda_api_sum.txt`, cudaMalloc/cudaFree are present; in the NVTX/kernels exports, the corresponding allocations fall in the setup span (model load / warm pool) — the finding (pass OR a documented real churn issue) is written in the report.
    - `grep -E "inference_3d_lowres|preprocess_3d_cascade_fullres" .planning/profiles/phase2/*nvtx*` (or the equivalent export) returns matches — per-config NVTX ranges are legible (INFR-005).
    - `grep -n "ERR_NVGPUCTRPERM" .planning/profiles/phase2/ncu_status.txt` returns 1 line; no ncu kernel metrics appear in any artifact under `.planning/profiles/phase2/`.
    - A stream-overlap statement (either observation) is present in the artifacts or the report (D-16 honesty).
    - Commit exists for the artifacts.
  </acceptance_criteria>
  <verify>cd /users/srv-mde/projects/monai-deploy-app-sdk && ls .planning/profiles/phase2/ && grep -c "cudaMalloc" .planning/profiles/phase2/phase2_bundle_cuda_api_sum.txt && grep -l "inference_3d_lowres" .planning/profiles/phase2/* 2>/dev/null</verify>
  <done>Phase 2 profiling artifacts for Phase 3 scoping are saved: RMM churn check, per-config NVTX legibility, remaining CPU-bound regions identified from the nsys kernel timeline, ncu documented unavailable, stream-overlap observed honestly (INFR-005/tasks 2.13-2.14).</done>
</task>

<task type="auto">
  <name>Task 3: 02-BENCHMARK-REPORT.md — the two-bar report with per-operator deltas and the D-18 deviation</name>
  <files>.planning/phases/2-gpu-acceleration/02-BENCHMARK-REPORT.md</files>
  <read_first>
    - .planning/benchmarks/phase2_results.csv (Task 1 output — the numbers)
    - .planning/benchmarks/baseline-2026-08-18.csv (Phase 1 per-operator: in-study 42.1 s, inference 27.2 s, etc.)
    - .planning/baseline_results.csv (reference 169,747 ± 7,274 ms + per-stage)
    - .planning/profiles/phase2/ (Task 2 findings to cite)
    - .planning/phases/2-gpu-acceleration/02-RESEARCH.md (Open Question 1 — the documented resolution to quote)
  </read_first>
  <action>
Write `.planning/phases/2-gpu-acceleration/02-BENCHMARK-REPORT.md` with EXACTLY these
sections (fill with the measured numbers):
1. **Scope & method** — fresh-process reps, 1 warmup + 3 measured, A100-SXM4-40GB,
   airway study, venv/date; both `HOLOSCAN_MODEL_LIST` scopes.
2. **Bar (a): same-scope fullres-only vs Phase 1 61.8 s** — mean ± std table (E2E +
   per-operator: preprocess/inference/postresample/ensemble/postprocess/write) with
   deltas vs `baseline-2026-08-18.csv`; explicit statement of whether the
   "any positive E2E improvement" bar (D-18) is MET at same scope, and where the win/loss
   comes from (CuPy preprocess vs scipy-resample round-trip).
3. **Bar (b): headline bundle vs 169.7 s reference** — mean ± std table (E2E + per-config
   operators) with the speedup ratio vs 169.7 s and per-operator deltas vs the reference
   per-stage numbers (setup ~12.8 s, inference ~138 s, postprocess 9–23 s, write ~1.2 s).
4. **Bundle vs 61.8 s (documented deviation from literal D-18)** — the bundle number
   next to 61.8 s WITH the scope note: "Phase 1's 61.8 s was single-config fullres; the
   bundle runs ~2x the inference work (3 configs). Comparing them is scope-asymmetric;
   the same-scope bar in §2 is the controlling positive-improvement bar. RESEARCH.md
   estimated a serial 3-config bundle at 70–80 s — measured: <X s>."
5. **Profiling summary** — cudaMalloc/cudaFree findings (RMM active or churn issue),
   per-config NVTX legibility, top CPU-bound regions from the kernel timeline (the
   Phase 3 shortlist — e.g. scipy resample spans, CC, writers), stream-overlap
   observation (D-16 honest note), ncu unavailable (ERR_NVGPUCTRPERM).
6. **Phase 3 handoff** — ranked bottleneck list with the trace file citations, plus
   carried items: INFR-02 cross-study buffer reuse (user adding reference examples),
   ≥5-CT corpus re-run, 2d model validation.
Commit the report.
  </action>
  <acceptance_criteria>
    - `.planning/phases/2-gpu-acceleration/02-BENCHMARK-REPORT.md` exists with all six numbered sections (grep each section heading).
    - §2 contains a delta column vs Phase 1 AND an explicit MET/NOT-MET sentence for the same-scope bar; §3 contains the `169.7`/`169,747` baseline and a speedup ratio number; §4 contains the string `D-18` and `70` (the 70–80 s serial estimate) and the measured bundle total.
    - §5 cites the exact artifact filenames from `.planning/profiles/phase2/` and contains `ERR_NVGPUCTRPERM`; §6 contains `INFR-02`.
    - Every number in the report is traceable to `phase2_results.csv` or a profiling export (no unmeasured claims).
    - Commit exists.
  </acceptance_criteria>
  <verify>cd /users/srv-mde/projects/monai-deploy-app-sdk && grep -n "D-18\|169,747\|ERR_NVGPUCTRPERM\|INFR-02" .planning/phases/2-gpu-acceleration/02-BENCHMARK-REPORT.md</verify>
  <done>TEST-007 is complete: absolute latency + speedup ratios vs BOTH baselines, quantified improvement statement, per-operator deltas, and the Phase 3 scoping input (bottleneck ranking with trace citations) — with the D-18 scope deviation documented rather than hidden.</done>
</task>

## Verification
- `phase2_results.csv` valid (verify block, Task 1) with both scopes, per-config columns,
  both speedups, mean ± std summaries.
- `.planning/profiles/phase2/` contains trace + cuda_api_sum + NVTX + kernel exports +
  `ncu_status.txt`; RMM churn check and stream-overlap note present.
- `02-BENCHMARK-REPORT.md` covers both bars, the D-18 deviation, profiling findings, and
  the Phase 3 handoff.
- Commits made (small, imperative).

## Success Criteria
- [ ] TEST-006: E2E + per-operator benchmark for the fast app (fresh process, warmup excluded, mean ± std)
- [ ] TEST-007: speedup ratios vs 169.7 s AND same-scope improvement vs 61.8 s, quantified in the roadmap-required CSV + report
- [ ] INFR-005: per-config NVTX ranges verified in a saved nsys trace; cudaMalloc churn check recorded
- [ ] ncu documented unavailable (no fake kernel metrics); stream overlap noted honestly (D-16)
- [ ] Phase 3 receives a ranked, trace-cited bottleneck list
