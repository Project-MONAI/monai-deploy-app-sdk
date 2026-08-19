---
phase: 3-optimization
plan: "05"
type: execute
wave: 5
depends_on: ["3-optimization-04"]
files_modified:
  - .planning/scripts/phase3_benchmark.py
  - .planning/benchmarks/phase3_results.csv
  - .planning/phases/03-optimization/03-BENCHMARK-REPORT.md
  - .planning/phases/03-optimization/03-GATE-RESULTS.json
autonomous: true
requirements: [TEST-006, TEST-007, TEST-01, TEST-002, TEST-003]

must_haves:
  truths:
    - "phase3_results.csv contains the 2x2 benchmark matrix (resample flag OFF/ON x serial/concurrent, D-18 convention) for both scopes (fullres same-scope, bundle headline), fresh-process reps, warmup excluded (1 warmup + 3 measured per cell), per-operator columns, CUDA device recorded per Pitfall 7"
    - "03-BENCHMARK-REPORT.md carries the D-18 two-bar convention: same-scope fullres vs 61.8 s (Phase 1) AND vs 57.14 s (Phase 2) + bundle headline vs 169.7 s reference, with per-operator deltas vs phase2_results.csv (the Phase 3 'before')"
    - "The full gate suite (phase2_gate.py: 4 config pixel gates + SR 0.1% + residency) re-run in the final shipping configuration, every row green, -> 03-GATE-RESULTS.json (completes the TEST-02/TEST-003 deliverables — automated pixel-diff tool + SR tolerance check — as part of every Phase 3 gate run)"
    - "The three external-dependency items are recorded as blocked-on-external, non-blocking (D-26): >=5-CT corpus re-run (TEST-01 final), ncu admin access (ERR_NVGPUCTRPERM), INFR-02 user reference examples — in the report AND flagged for VERIFICATION.md"
    - "Deferred-with-reason decisions documented in the report: ACCEL-01/02/03 (ncu admin-blocked), MEM-01 (not a measured bottleneck — models load once), MEM-02 (8 GB target hardware-unverifiable on A100-40GB), pylibraft CC (evaluated, not taken — CuPy path adequate under D-20 trim)"
    - "TEST-01/TEST-006/TEST-007 met on the dev corpus with deviations recorded (D-26)"
  artifacts:
    - path: ".planning/scripts/phase3_benchmark.py"
      provides: "2x2 matrix harness (phase2_benchmark.py extended: cell = HOLOSCAN_CONCURRENT_FRAGMENTS x HOLOSCAN_GPU_RESAMPLE; device column; per-cell summary rows)"
    - path: ".planning/benchmarks/phase3_results.csv"
      provides: "Phase 3 benchmark record: cells x (1 warmup + 3 measured) x 2 scopes + summary rows"
    - path: ".planning/phases/03-optimization/03-BENCHMARK-REPORT.md"
      provides: "Two-bar report + per-operator deltas + optimization evidence rollup (D-21 overlap, MEM-003 delta, INFR-02 proof, GPU-resample verdict) + deferred/external-dependency sections"
    - path: ".planning/phases/03-optimization/03-GATE-RESULTS.json"
      provides: "Final shipping-configuration gate suite (4 pixel gates + SR + residency)"
  key_links:
    - from: ".planning/scripts/phase3_benchmark.py"
      to: ".planning/benchmarks/phase2_results.csv"
      via: "per-cell per-operator delta tables (same columns; the Phase 2 'before')"
    - from: "03-BENCHMARK-REPORT.md"
      to: "VERIFICATION.md (written after this phase)"
      via: "the D-26 external-dependency section is written to be lifted verbatim"
    - from: "03-GATE-RESULTS.json"
      to: "phase2_gate.py --report (existing arg)"
      via: "final shipping-configuration run (all four plan flags in their shipped states)"

user_setup: []
---

<objective>
Phase 3 close-out: the 2x2 benchmark matrix vs Phase 2 (D-18 two-bar convention), the final
full gate suite in the shipping configuration, and the 03-BENCHMARK-REPORT.md with every
optimization's measured evidence + the deferred/external-dependency record for VERIFICATION.md.

Purpose: ROADMAP 3.9/3.10 (TEST-006/TEST-007/TEST-01/TEST-002/TEST-003) — prove the measured
improvement over Phase 2 (bundle 129.54 s / fullres 57.14 s) on the same hardware and corpus,
with pixel-exact equivalence maintained (D-25). No new code under test — only the benchmark
harness, the final gate run, and the report.

Output: phase3_benchmark.py + phase3_results.csv, 03-GATE-RESULTS.json, 03-BENCHMARK-REPORT.md.
</objective>

<execution_context>
@~/.claude/get-shit-done/workflows/execute-plan.md
@~/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/ROADMAP.md (Phase 3 acceptance criteria + D-18 two-bar definitions)
@.planning/phases/03-optimization/03-CONTEXT.md (D-18 carry-over convention; D-25; D-26)
@.planning/phases/03-optimization/3-optimization-01-SUMMARY.md (concurrency flag state + overlap evidence paths)
@.planning/phases/03-optimization/3-optimization-02-SUMMARY.md (MEM-003 delta evidence path)
@.planning/phases/03-optimization/3-optimization-03-SUMMARY.md (INFR-02 proof evidence path)
@.planning/phases/03-optimization/3-optimization-04-SUMMARY.md (GPU-resample verdict — determines whether the matrix's ON column is run or recorded N/A)
@.planning/benchmarks/phase2_results.csv (the Phase 3 'before' — columns to diff against)
@.planning/phases/02-gpu-acceleration/02-BENCHMARK-REPORT.md (D-18 convention + report structure to mirror; the ranked §6 list each optimization must be scored against)
@.planning/scripts/phase2_benchmark.py (extend, do not rebuild — D-25)

<interfaces>
<!-- phase2_benchmark.py structure (verified): -->
- run_once(study_dir, model, output_root, python, scope): fresh subprocess, env from os.environ
  copy, HOLOSCAN_MODEL_LIST pinned per scope (fullres = "3d_fullres"; bundle = UNSET), parses
  the app's config-tagged `timing:` JSON logs + the single app-level study_timing_summary.
- summarize_numeric(rows, header): mean±std summary row (warmups excluded).
- CSV: scope,study,rep,warmup,total_ms + per-config operator columns; P1_CENTER_MS = 61800.0,
  reference 169,747 ms constants.
- Invocation convention: ulimit -s unlimited; 32 MB stack rlimit inside; fresh process per rep.
Phase 3 extension (new script phase3_benchmark.py, copy + minimal changes):
- add `--cell` or auto-iterate the matrix: for scope in {fullres, bundle}: for
  cell in {(concurrent, resample)}: 1 warmup + 3 measured reps.
- env per cell: HOLOSCAN_CONCURRENT_FRAGMENTS=1/0 (explicit — never rely on defaults) and
  HOLOSCAN_GPU_RESAMPLE=1/ (unset). The resample-ON cells run ONLY if Plan 04 shipped the flag
  as byte-identical opt-in; otherwise record those cells as "NA (resample flag not byte-identical — see verdict)" rows, not blank space.
- add a `cell` column (e.g. "conc-resample-off") and a `gpu` column (CUDA_VISIBLE_DEVICES value — Pitfall 7 provenance).
- per-cell mean±std summary rows (warmups excluded), same as Phase 2.
Expected runtime: bundle cell ≈ 4 reps x ~110-130 s ≈ 9 min; fullres cell ≈ 4 x ~55 s ≈ 4 min;
4 cells x 2 scopes ≈ 45-55 min wall. Run it; it is the phase's evidence.
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: phase3_benchmark.py 2x2 matrix + run -> phase3_results.csv</name>
  <files>.planning/scripts/phase3_benchmark.py, .planning/benchmarks/phase3_results.csv</files>
  <read_first>
    - .planning/scripts/phase2_benchmark.py (FULL — the copy base; its header docstring defines the D-18 scopes and the CSV contract)
    - .planning/phases/03-optimization/3-optimization-04-SUMMARY.md (resample flag state — decides ON-column runnability)
    - .planning/phases/03-optimization/03-RESEARCH.md — Measurement Architecture section (the 2x2 extension definition) + Pitfall 6 (timing record order nondeterministic under concurrency — phase2_benchmark.py parses BY OPERATOR NAME so it is safe; do not add order-dependent asserts) + Pitfall 7 (device pinning)
  </read_first>
  <action>
    1. Create `.planning/scripts/phase3_benchmark.py` by copying `phase2_benchmark.py` and extending per the interfaces block: matrix iteration over (scope x cell), explicit env for both flags per cell, `cell` + `gpu` CSV columns, per-cell summary rows, resample-ON cells → "NA" rows with the reason when the Plan-04 verdict is OFF. Keep the fresh-process/warmup-excluded/methodology docstring accurate; keep the parse-by-operator-name (Pitfall 6-safe).
    2. Pin a free GPU (Pitfall 7) and run the FULL matrix: `/tmp/monai-env/.venv/bin/python .planning/scripts/phase3_benchmark.py --out .planning/benchmarks/phase3_results.csv` (design the CLI so one invocation runs the whole matrix; record the device in every row).
    3. Sanity-check the CSV before reporting: every non-NA cell has 1 warmup + 3 measured rows; totals are in the plausible band (bundle ~90-130 s; fullres ~45-60 s — if a cell regresses by >20% vs Phase 2, do NOT hide it: flag it in the report as a measured regression with the per-operator breakdown pointing at the cause).
    Commit after this task (script + CSV).
  </action>
  <verify>
    <automated>test -f .planning/scripts/phase3_benchmark.py && test -f .planning/benchmarks/phase3_results.csv && /tmp/monai-env/.venv/bin/python -c "
import csv
rows = list(csv.DictReader(open('.planning/benchmarks/phase3_results.csv')))
meas = [r for r in rows if r.get('warmup') not in ('1','summary') and r.get('rep') not in ('mean±std',)]
cells = {(r['scope'], r['cell']) for r in meas if r.get('cell')}
assert all(len([m for m in meas if (m['scope'], m.get('cell')) == c]) >= 3 for c in cells), 'a cell lacks 3 measured reps'
print('cells:', sorted(cells))
print('csv ok')
"</automated>
  </verify>
  <acceptance_criteria>
    - phase3_results.csv exists; every runnable (scope, cell) has >= 3 measured reps + 1 warmup; summary rows present per cell; `gpu` column populated; NA resample-ON cells carry the reason string (if applicable).
    - The script parses timing by operator name (no record-order assumptions — Pitfall 6).
  </acceptance_criteria>
  <done>The Phase 3 2x2 matrix is measured with the Phase 2 methodology, device-pinned, and committed — the raw evidence for the two-bar report.</done>
</task>

<task type="auto">
  <name>Task 2: Final gate suite (shipping config) + 03-BENCHMARK-REPORT.md + close-out</name>
  <files>.planning/phases/03-optimization/03-GATE-RESULTS.json, .planning/phases/03-optimization/03-BENCHMARK-REPORT.md</files>
  <read_first>
    - .planning/scripts/phase2_gate.py (final invocation with the existing --report arg targeting 03-GATE-RESULTS.json)
    - .planning/benchmarks/phase3_results.csv + .planning/benchmarks/phase2_results.csv (the delta tables' inputs)
    - .planning/phases/02-gpu-acceleration/02-BENCHMARK-REPORT.md (mirror the structure: two bars, per-operator deltas, honest deviations)
    - All four 3-optimization-*-SUMMARY.md + the evidence files (.planning/profiles/phase3/overlap.md, rmm_openq1.md; evidence/mem003_vram.md; evidence/infr02_proof.md; evidence/gpu_resample_verdict.md)
  </read_first>
  <action>
    1. Final gate suite in the SHIPPING configuration (all four plan flags in their shipped states — concurrency default per Plan 01, release hook active per Plan 02, caches active per Plan 03, resample flag at its shipped default per Plan 04; pinned GPU, device recorded): `/tmp/monai-env/.venv/bin/python .planning/scripts/phase2_gate.py --report .planning/phases/03-optimization/03-GATE-RESULTS.json` — every row must be green (fullres 99.99986%/3 documented fp16<->fp32 boundary class, lowres/cascade/bundle 100.00000%/0, SR 0.0% vs the 0.1% bar, residency PASS). If the shipping concurrency default is ON, the JSON note records it. This run also stands as the TEST-02 (SR 0.1%) and TEST-003 (automated pixel-diff fails on divergence) deliverable for the phase.
    2. Write `03-BENCHMARK-REPORT.md` (sections, mirroring 02-BENCHMARK-REPORT.md):
       §1 Scope & method (2x2 matrix, D-18 convention, device, corpus = single airway dev study, flags shipped).
       §2 Same-scope bar: best shipping cell fullres E2E mean±std vs 61.8 s (Phase 1) AND vs 57.14 s (Phase 2) — speedup ratios both ways; per-operator delta table vs phase2_results.csv columns.
       §3 Headline bar: best shipping cell bundle E2E vs 169.747 s reference; per-operator deltas; the scope-asymmetry note (bundle-vs-61.8 s is printed, never a bar — D-18).
       §4 The 2x2 matrix in full (all cells, incl. NA rows with reasons) — the concurrency effect (vs Phase 2's serial baseline = the Phase 2 numbers) and the resample-flag effect (if ON ran) isolated per cell.
       §5 Optimization evidence rollup, each vs its §6-Phase-2 bottleneck item: D-21 (overlap.md citation + measured E2E effect), MEM-003 (pool/driver delta from mem003_vram.md — memory deliverable, no speed claim), INFR-02 (proof summary from infr02_proof.md — allocator-traffic deliverable, no speed claim), GPU resample (verdict + if ON: the resample-span collapse measured in the matrix).
       §6 Deviations & honesty: any regressed cell, the RMM Open-Q1 outcome, the driver-level-flat MEM-003 result if that is what measured, the resample fallback if taken.
       §7 Deferred-with-reason (for VERIFICATION.md, verbatim-liftable): ACCEL-01/02/03 — ncu admin-blocked (ERR_NVGPUCTRPERM) + hostile compile env; MEM-01 — not a measured bottleneck (models load once, bootstrap amortized); MEM-02 — 8 GB target hardware-unverifiable on A100-40GB; pylibraft CC (ROADMAP 3.5) — evaluated, not taken: postprocess 9.9 s ≈ 7.6% of bundle; CuPy CC path adequate, pylibraft not warranted under D-20 trim.
       §8 External dependencies (blocked-on-external, non-blocking, D-26 — verbatim-liftable for VERIFICATION.md): (1) >=5-CT corpus re-run (TEST-01 final gate) — blocked on CT data; (2) ncu kernel profiling — blocked on admin access; (3) INFR-02 user reference examples — user adding during Phase 3. Each states: re-opens as a gap plan if the dependency lands.
       Requirement status line: TEST-01/TEST-002/TEST-003/TEST-006/TEST-007 met on the dev corpus with the §7/§8 deviations; TEST-01's >=5-CT half = external dependency (1).
    3. Close-out commit: report + final gate JSON + CSV (if not already committed).
  </action>
  <verify>
    <automated>test -f .planning/phases/03-optimization/03-BENCHMARK-REPORT.md && test -f .planning/phases/03-optimization/03-GATE-RESULTS.json && /tmp/monai-env/.venv/bin/python -c "
import json
d = json.load(open('.planning/phases/03-optimization/03-GATE-RESULTS.json'))
assert d.get('all_gates_pass') is True, 'final gate not green (schema: top-level all_gates_pass; per-gate pass)'
t = open('.planning/phases/03-optimization/03-BENCHMARK-REPORT.md').read()
for s in ['169,747\|169.7', '61.8', '57.14', 'ACCEL-01', 'MEM-01', 'MEM-02', 'ncu', 'corpus', 'reference examples']:
    assert any(x in t for x in s.split('|')), f'report missing section content: {s}'
print('final gate green + report sections present')
"</automated>
  </verify>
  <acceptance_criteria>
    - 03-GATE-RESULTS.json fully green in the shipping configuration.
    - 03-BENCHMARK-REPORT.md contains ALL eight §-sections; both bars computed with mean±std and speedup ratios; per-operator delta tables reference the exact phase2_results.csv column names; every Phase 3 optimization is scored against its §6-Phase-2 bottleneck item.
    - §7/§8 are self-contained (a VERIFICATION.md author can lift them without reading the phase).
    - No number in the report appears without its source artifact path cited inline.
  </acceptance_criteria>
  <done>Phase 3 is closeable: the measured 2x2 evidence is committed, the final gate is green, and the report tells the two-bar story with every deferred decision and external dependency recorded for VERIFICATION.md.</done>
</task>

</tasks>

<verification>
- phase3_results.csv: full matrix (or documented NA cells), device-pinned, warmup-excluded summaries.
- 03-GATE-RESULTS.json green; 03-BENCHMARK-REPORT.md complete per the 8-section spec.
- ROADMAP Phase 3 acceptance: profiling-citation-per-optimization (each §5 row cites its trace/evidence), pixel-exact after optimizations (§ final gate), improvement over Phase 2 quantified, CSV at .planning/benchmarks/phase3_results.csv, per-v2-requirement decision documented (§7).
</verification>

<success_criteria>
- Both D-18 bars reported honestly (same-scope vs Phase 1 AND Phase 2; bundle headline vs reference; scope asymmetry noted, never massaged).
- Every optimization's effect is measured, not estimated (or the null result is reported as such).
- The phase closes with zero open questions that block VERIFICATION.md — only the three recorded external dependencies.
</success_criteria>

<output>
After completion, create `.planning/phases/03-optimization/3-optimization-05-SUMMARY.md` covering: the two-bar numbers, the 2x2 matrix headline, final gate outcome, the report's deviation list, commits, and the explicit handoff note for /gsd-verify-phase (what VERIFICATION.md should check, and that §7/§8 are liftable verbatim).
</output>
