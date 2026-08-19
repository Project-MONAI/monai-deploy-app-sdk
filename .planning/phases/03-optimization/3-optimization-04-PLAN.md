---
phase: 3-optimization
plan: "04"
type: execute
wave: 4
depends_on: ["3-optimization-03"]
files_modified:
  - examples/apps/cchmc-nnunet-fast/my_app/operators/gpu_zoom.py
  - examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py
  - examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py
  - examples/apps/cchmc-nnunet-fast/scripts/test_gpu_zoom.py
autonomous: true
requirements: [GPUP-01, GPUP-02, TEST-01, TEST-002, TEST-003]

must_haves:
  truths:
    - "gpu_zoom_grid_mode is a CuPy RawKernel that replicates scipy 1.15.3 NI_GeometricTransform zoom(grid_mode=True, mode='nearest') per-voxel arithmetic: double accumulation, exact C-order tap iteration (last axis fastest), per-tap nearest clamping, --fmad=false (or explicit __dmul_rn/__dadd_rn), __double2float_rn output"
    - "test_gpu_zoom.py: np.array_equal vs scipy on ALL real call-site shapes (256³→(255,256,255) o3; 256³→(201,202,201) o3; 256³→(255,256,255) o1 seg multihot 0/1; (255,256,255)→256³ o1 ×2ch prob) + randomized shapes/orders 0–3 (up and down zoom)"
    - "HOLOSCAN_GPU_RESAMPLE defaults OFF; the OFF path is byte-for-byte Phase 2/3 behavior (gate re-run both ways)"
    - "Flag ON: all 4 pixel-exact gates + SR + residency PASS (byte-identity achieved) — OR any failure keeps the flag OFF, the measured divergence is documented, and the phase still ships (D-22 fallback, GPUP-01/02 met-with-deviation by the gate outcome)"
    - "GPUP-02 (zero CPU-GPU transfers in the resample span) holds for the image path only when the flag is ON; the numpy mean/std reductions stay CPU by Phase 1 decision and are recorded as the residual"
  artifacts:
    - path: "examples/apps/cchmc-nnunet-fast/my_app/operators/gpu_zoom.py"
      provides: "gpu_zoom_grid_mode(cupy_fp32, zoom_factors, order) -> cupy_fp32 + per-axis double splvals (3 × out_dim × (order+1))"
    - path: "examples/apps/cchmc-nnunet-fast/scripts/test_gpu_zoom.py"
      provides: "The byte-identity arbiter: np.array_equal vs scipy, real + random shapes/orders"
    - path: ".planning/phases/03-optimization/gates/03-GATE-resample-off.json"
      provides: "Flag-OFF regression (must equal Phase 2/3 behavior)"
    - path: ".planning/phases/03-optimization/gates/03-GATE-resample-on.json"
      provides: "Flag-ON full gate (only valid if byte-identity held)"
    - path: ".planning/phases/03-optimization/evidence/gpu_resample_verdict.md"
      provides: "ON or OFF verdict with the per-tensor identity table (or measured-divergence table if the fallback shipped) + GPUP-02 residual note"
  key_links:
    - from: "preprocess_operator._resample_to_shape (non-separate-z branch)"
      to: "gpu_zoom_grid_mode"
      via: "flag check → gpu_zoom(cp.asarray(data[c]), [n/o per axis], order) replacing skimage resize"
    - from: "preprocess_operator._resize_segmentation (multihot path)"
      to: "gpu_zoom_grid_mode"
      via: "o1 float mask → gpu_zoom → >=0.5 threshold (the .astype(tpe) tail preserved)"
    - from: "postresample_operator.resample_probabilities_to_shape (non-separate-z branch)"
      to: "gpu_zoom_grid_mode"
      via: "flag check → per-channel gpu_zoom at o1"
    - from: "the separate-z map_coordinates branches"
      to: "scipy (UNCHANGED)"
      via: "never active in this bundle; order_z=0 CuPy was byte-identical anyway — no port needed"

user_setup: []
---

<objective>
Run the D-22 gated experiment: a CuPy RawKernel GPU resampler that is byte-identical to the
scipy.ndimage.zoom(grid_mode=True) reference path, behind HOLOSCAN_GPU_RESAMPLE (default OFF).

Purpose: Bottleneck #1 in `02-BENCHMARK-REPORT.md` §6 — the scipy resample spans are ~28.8 s of
the 129.5 s bundle (22.2%), 99.9% CPU. Research verdicts (measured): stock
cupyx.scipy.ndimage.map_coordinates is byte-identical at order 0 only; order ≥1 requires
replicating scipy's own per-voxel double arithmetic with exact accumulation order and
--fmad=false; monai.data.Resample is dead by construction. All three active call sites reduce to
ONE primitive (zoom, grid_mode=True, mode='nearest'). This is a GATED EXPERIMENT per D-22 — if
byte-identity fails after good-faith effort, the flag stays OFF, divergence is documented, and
the phase still ships.

Output: gpu_zoom.py kernel, the identity test suite, flag plumbing at all three call sites, gate
suite both ways, and the verdict record.
</objective>

<execution_context>
@~/.claude/get-shit-done/workflows/execute-plan.md
@~/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/ROADMAP.md
@.planning/phases/03-optimization/03-CONTEXT.md (D-22 locked — gated experiment, default OFF)
@.planning/phases/03-optimization/03-RESEARCH.md — §D-22 Domain READ FULLY before writing the kernel: the call-site table (real shapes/orders), the byte-identity verdict table, the 6-point scipy-faithful kernel spec (scipy 1.15.3 ni_interpolation.c NI_GeometricTransform), Pitfall 3 (FMA: RawKernel options is a TUPLE) and Pitfall 4 (scipy computes in double)
@.planning/phases/03-optimization/3-optimization-03-SUMMARY.md (buffer-cache state in the two operators — the flag branches go NEXT TO the cached allocations, do not disturb them)
@.planning/scripts/probes-phase3/probe_resample_identity.py (the research's live identity probes — the numerical ground truth for the test expectations)

# Files under change:
@examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py (_resample_to_shape ~211–287; _resize_segmentation ~287–309; _resample_seg_to_shape ~310–420)
@examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py (resample_probabilities_to_shape ~98–171; the separate-z map_coordinates branch ~163)

<interfaces>
<!-- The kernel spec (research §D-22, verified against scipy 1.15.3 C source). -->
For zoom(zoom_factors, order, mode='nearest', grid_mode=True), per axis, per output index kk:
1. Coordinate: cc = kk + 0.5; cc *= zoom; cc -= 0.5  (zoom = out_dim/in_dim; identical to the
   app's existing map_rows formula). nearest clamp: cc < 0 → 0; cc > len_in−1 → len_in−1.
2. Spline weights splvals[axis][kk][0..order] computed IN DOUBLE (get_spline_interpolation_weights,
   ni_splines.c closed-form B-spline). Tap start: order odd → floor(cc) − order/2; even →
   floor(cc + 0.5) − order/2. Precompute per axis on GPU (tiny: 3 × out_dim × (order+1) doubles).
3. Per-voxel accumulation — EXACT ORDER: t = 0.0 (double); iterate the (order+1)^ndim tap grid
   ff in C ORDER (last axis fastest): coeff = (double) input[idx] (fp32 upcast); for ll in
   0..ndim−1: coeff *= splvals[ll][ff[ll]] (each multiply is a separate RN op); t += coeff
   (separate RN add). Every tap index is clamped per axis (nearest) BEFORE the input lookup.
4. out = (float32) t via __double2float_rn.
5. NO FMA contraction: compile with options=("--fmad=false",) — TUPLE, not list (list raises
   TypeError) — or explicit __dmul_rn/__dadd_rn for every double op.
6. API: gpu_zoom_grid_mode(x: cp.ndarray fp32 3D C-contiguous, zoom_factors: Sequence[float],
   order: int) -> cp.ndarray fp32. splvals precomputed per (shape, zooms, order) call (tiny).
fp64 A100 throughput is fine: 256³ × 27 taps ≈ 2 GFLOP « 1 s.
Flag helper (shared, put in gpu_zoom.py or config):
    def gpu_resample_enabled() -> bool: return os.environ.get("HOLOSCAN_GPU_RESAMPLE") == "1"
Default OFF = the skimage resize / scipy paths run exactly as today (byte-for-byte Phase 2/3).
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: scipy-faithful RawKernel (gpu_zoom.py) + byte-identity suite (test_gpu_zoom.py)</name>
  <files>
    examples/apps/cchmc-nnunet-fast/my_app/operators/gpu_zoom.py,
    examples/apps/cchmc-nnunet-fast/scripts/test_gpu_zoom.py
  </files>
  <read_first>
    - .planning/phases/03-optimization/03-RESEARCH.md — §D-22 kernel spec (all 6 points) + the byte-identity verdict table + Pitfall 3 (FMA/options tuple) + Pitfall 4 (double accumulation)
    - .planning/scripts/probes-phase3/probe_resample_identity.py (measured expectations: o0 byte-equal, o1/o3 stock-CuPy NOT equal; scipy-fp32 == scipy-fp64-cast-back byte-equal)
    - examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py — _resample_to_shape's non-sep-z branch (the exact zoom factor computation: output/input per axis) and _resize_segmentation (the o1 multihot + clip + >=0.5 semantics the kernel must serve)
  </read_first>
  <action>
    1. Create `my_app/operators/gpu_zoom.py` per the interfaces spec: `gpu_zoom_grid_mode(x, zoom_factors, order)` + `gpu_resample_enabled()`. Implementation notes:
       - splvals: compute per axis with a small CuPy double kernel (or numpy double) replicating get_spline_interpolation_weights; the END-TO-END byte-equality test is the arbiter — if splvals computation diverges, the o1/o3 tests will catch it.
       - main kernel: 1 thread per output voxel; the (order+1)^ndim tap loop is a fixed unroll-free C-order loop over the flattened tap index (last axis fastest — decompose the flat index with the same strides scipy uses); clamp each tap coordinate per axis before indexing; accumulate with __dmul_rn/__dadd_rn AND pass options=("--fmad=false",) (belt and braces; options is a TUPLE).
       - Support ndim=3 only (all active call sites are 3D) — document the restriction; raise NotImplementedError for other ndim.
       - Inputs/outputs: cp fp32, C-contiguous (assert `x.flags.c_contiguous`); dtype other than fp32 → raise (the call sites are all fp32; the seg path casts to float first, exactly like the scipy path does).
    2. Create `scripts/test_gpu_zoom.py` — the D-22 arbiter. Cases (ALL `np.array_equal` vs `scipy.ndimage.zoom(..., mode='nearest', grid_mode=True)` computed on the same CPU numpy input; any failure → nonzero exit + a printed divergence table: n_diff/total, max_abs, first-10 differing coordinates, per case):
       - REAL call-site shapes (from the research table): (a) 256³→(255,256,255) order 3; (b) 256³→(201,202,201) order 3; (c) 256³→(255,256,255) order 1 on 0/1 multihot masks (seg path); (d) (255,256,255)→256³ order 1, 2 channels (probabilities path).
       - RANDOMIZED: 8 seeded random cases — shapes in 128–320 per axis, zooms both up and down (0.7–1.3), orders 0/1/3, fp32 random data; plus one degenerate: zoom factor exactly 1.0 (identity) and one non-integer output dim mix.
       - ORDER-0 sanity: byte-equal is also the stock-CuPy baseline (proves the test harness itself is sound before trusting the o1/o3 verdicts).
       - Timing note (not a gate): print per-call ms for case (a) — expected «1 s (fp64 A100 budget from the research).
    3. RED→GREEN: run the suite first with the kernel unimplemented (import error expected), then iterate until exit 0. If, after a good-faith full effort (the spec is precise — tap order, clamp, FMA are the three known traps), any case still diverges: do NOT ship the flag; write the divergence table into `.planning/phases/03-optimization/evidence/gpu_resample_verdict.md` under "## Fallback: flag stays OFF" and STOP Task 2's gate-ON half (still do the Task 2 OFF regression + wiring so the flag exists but defaults OFF).
    Commit after this task (kernel + tests, even in the fallback state).
  </action>
  <verify>
    <automated>cd examples/apps/cchmc-nnunet-fast && ulimit -s unlimited && /tmp/monai-env/.venv/bin/python scripts/test_gpu_zoom.py; test $? -eq 0</automated>
  </verify>
  <acceptance_criteria>
    - `test_gpu_zoom.py` exits 0 with all real + random cases byte-equal (or the fallback verdict file exists with the divergence table and the flag is confirmed OFF-only).
    - `grep -c "fmad=false" my_app/operators/gpu_zoom.py` ≥ 1 and `grep -c "__double2float_rn" my_app/operators/gpu_zoom.py` ≥ 1.
    - The kernel is compiled with options as a TUPLE (grep the RawKernel construction).
    - No change to any operator file yet (wiring is Task 2) — this task is kernel + tests only.
  </acceptance_criteria>
  <done>The GPU resample math is byte-identical to scipy at every real call-site shape and under randomization — or the fallback is documented with measured divergence and the flag is provably OFF-only.</done>
</task>

<task type="auto">
  <name>Task 2: Flag plumbing at the 3 call sites + gate suite both ways + verdict record</name>
  <files>
    examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py,
    examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py,
    .planning/phases/03-optimization/evidence/gpu_resample_verdict.md,
    .planning/phases/03-optimization/gates/
  </files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py — the three scipy sites: _resample_to_shape non-sep-z branch (~254–280), _resize_segmentation (~287–309), the separate-z branch (leave scipy)
    - examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py — resample_probabilities_to_shape non-sep-z branch (~144–170)
    - .planning/phases/03-optimization/03-RESEARCH.md — §D-22 "Flag placement" + Pattern 2 (the exact if/else shape)
    - .planning/scripts/phase2_gate.py (--json-out from Plan 01; env pass-through verified in Plan 01 Task 2)
  </read_first>
  <action>
    1. Wire `gpu_resample_enabled()` (from gpu_zoom.py) at the three non-separate-z call sites, research Pattern 2 shape:
       - `_resample_to_shape` non-sep-z branch: per channel `zooms = [float(n)/float(o) for o, n in zip(data[c].shape, new_shape)]` (match the EXACT factor direction the scipy/skimage path uses — verify against the existing map_rows formula, not the pattern snippet); flag ON → `out = gpu_zoom_grid_mode(cp.asarray(data[c]), zooms, order)`; OFF → the existing `resize(...)` line UNCHANGED.
       - `_resize_segmentation`: flag ON → per-label float mask through `gpu_zoom_grid_mode(mask_cp, zooms, order)` then the SAME threshold/clip/cast tail the scipy path has (`>= 0.5`, `.astype(tpe)`); the non-multihot single-label early return (~line 301) likewise gets the flag branch (or, if the shape is 0/1 through o1, the same call).
       - `resample_probabilities_to_shape` non-sep-z branch: flag ON → per-channel gpu_zoom at the existing order; OFF unchanged.
       - The separate-z `map_coordinates` branches: NO flag (inactive in this bundle; scipy stays — documented in a one-line comment).
       - Interaction with Plan 03's buffer cache: the flag-ON path keeps buffers on GPU end-to-end (GPUP-02 for the span) — use the cache for the output buffer where the site already caches it; the flag-OFF path is byte-for-byte the current code (the D2H/H2D round trip stays).
       - GPUP-02 residual (record in the verdict): numpy mean/std reductions stay CPU (Phase 1 bit-exactness decision) + the ~8 MB mask round trip — GPUP-02 is met for the resample span only; the full requirement wording (zero transfers in ALL preprocessing) is met-with-this-residual and the verdict says so.
    2. Gate suite BOTH ways (D-25; pinned GPU, device recorded; all prior plans in shipping state):
       - OFF: `/tmp/monai-env/.venv/bin/python .planning/scripts/phase2_gate.py --json-out .planning/phases/03-optimization/gates/03-GATE-resample-off.json` — must equal Phase 2/3 results exactly (fullres 99.99986%/3 boundary class, others 100.00000%/0, SR 0.0%, residency PASS) — this proves the plumbing changed nothing by default.
       - ON (ONLY if Task 1 was byte-identical): `HOLOSCAN_GPU_RESAMPLE=1 .../phase2_gate.py --json-out .planning/phases/03-optimization/gates/03-GATE-resample-on.json` — all 4 pixel gates + SR + residency must pass. Any divergence here (even with unit tests green) → flag stays OFF, divergence documented (end-to-end ≠ unit: the Phase 1 lesson).
    3. `.planning/phases/03-optimization/evidence/gpu_resample_verdict.md`: verdict (SHIPPED-ON as opt-in flag / OFF-with-divergence), the per-tensor identity table from Task 1, both gate JSON summaries, the GPUP-02 residual paragraph, and the expected-latency note for Plan 05's 2×2 matrix (if ON: expect the ~22.2 s resample spans to collapse to seconds — the matrix will measure it; if OFF: the matrix's ON column is recorded as N/A-not-byte-identical).
    4. The flag DEFAULT STAYS OFF in both outcomes (D-22: "default OFF = scipy"). Plan 05's 2×2 matrix exercises ON as the experimental cell.
    Commit after this task.
  </action>
  <verify>
    <automated>grep -q "gpu_resample_enabled" examples/apps/cchmc-nnunet-fast/my_app/operators/preprocess_operator.py && grep -q "gpu_resample_enabled" examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py && test -f .planning/phases/03-optimization/gates/03-GATE-resample-off.json && test -f .planning/phases/03-optimization/evidence/gpu_resample_verdict.md && /tmp/monai-env/.venv/bin/python -c "import json; d=json.load(open('.planning/phases/03-optimization/gates/03-GATE-resample-off.json')); print('off-gate ok' if all((g.get('ok') or g.get('status')=='PASS') for g in d['gates'] if isinstance(g, dict)) else 'GATE FAIL')"</automated>
  </verify>
  <acceptance_criteria>
    - Exactly 3 flag sites (grep `gpu_resample_enabled` call sites: preprocess 2 — image + seg — and postresample 1; the helper definition itself is in gpu_zoom.py); every OFF branch is the verbatim existing code.
    - `03-GATE-resample-off.json` fully green (plumbing regression-proof).
    - `03-GATE-resample-on.json` exists AND fully green (byte-identity path) — or the verdict file records the fallback with the measured divergence and no ON gate JSON is claimed.
    - `gpu_resample_verdict.md` contains the verdict line, identity table, GPUP-02 residual, and the Plan-05 matrix note.
  </acceptance_criteria>
  <done>GPUP-01 is delivered as the D-22 gated experiment in its final state: byte-identical opt-in flag (ON provable at the pixel level), or documented divergence with the flag safely OFF — either way the phase ships and the OFF path is regression-proof.</done>
</task>

</tasks>

<verification>
- `test_gpu_zoom.py` exit 0 (or fallback documented — the phase is NOT blocked by a failed experiment, D-22).
- `03-GATE-resample-off.json` green in all outcomes; `03-GATE-resample-on.json` green iff the flag is claimable.
- The separate-z scipy branches are untouched; the Plan-03 buffer cache and D-13 comments are intact.
- Flag default OFF in the committed code.
</verification>

<success_criteria>
- One GPU implementation of zoom(grid_mode=True, mode='nearest', order 0/1/3) covers every active call site — verified byte-identical per tensor and per config.
- The experiment is decision-complete: a single verdict file states ON-as-opt-in or OFF-with-divergence, with the evidence tables.
- No default-behavior change (OFF = byte-for-byte Phase 2/3), proven by the OFF gate run.
</success_criteria>

<output>
After completion, create `.planning/phases/03-optimization/3-optimization-04-SUMMARY.md` covering: the kernel implementation choices (splvals computation path, tap-loop structure), the identity test results (all cases + any fallback divergence table), the flag state shipped, both gate JSONs' outcomes, the measured per-call kernel timing, commits, and deviations.
</output>
