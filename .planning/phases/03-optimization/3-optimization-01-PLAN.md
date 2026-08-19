---
phase: 3-optimization
plan: "01"
type: execute
wave: 1
depends_on: []
files_modified:
  - examples/apps/cchmc-nnunet-fast/my_app/app.py
  - examples/apps/cchmc-nnunet-fast/my_app/gpu_bootstrap.py
  - .planning/scripts/phase2_gate.py
autonomous: true
requirements: [TEST-01, TEST-002, TEST-003]

must_haves:
  truths:
    - "Concurrent independent-fragment execution (D-21) is wired as EventBasedScheduler in app.py compose(), gated by HOLOSCAN_CONCURRENT_FRAGMENTS, with the flag-OFF path byte-for-byte Phase 2 serial behavior"
    - "RMM initial-pool reservation is re-verified against the live venv (research Open Q1 / Pitfall 8 environment drift) and either pinned via initial_pool_size or documented with the measured size"
    - "All 4 pixel-exact config gates + SR 0.1% + residency PASS with concurrency enabled (D-25 anchor), OR the measured ceiling is documented with a trace citation and the flag default stays OFF (D-21 fallback clause)"
    - "An nsys trace of one concurrent bundle run shows overlapping per-config NVTX spans (vs Phase 2's single-thread back-to-back), recorded as the trace citation"
  artifacts:
    - path: "examples/apps/cchmc-nnunet-fast/my_app/app.py"
      provides: "EventBasedScheduler wiring (worker_thread_number=5) + HOLOSCAN_CONCURRENT_FRAGMENTS flag"
    - path: ".planning/profiles/phase3/"
      provides: "Open-Q1 RMM reservation evidence + concurrent bundle nsys trace (rep + sqlite + stats exports) + overlap note"
    - path: ".planning/phases/03-optimization/gates/03-GATE-concurrent.json"
      provides: "Full 4-config gate suite results with concurrency ON"
    - path: ".planning/phases/03-optimization/gates/03-GATE-serial.json"
      provides: "Flag-OFF regression results (must match Phase 2 behavior)"
  key_links:
    - from: "my_app/app.py compose() tail (after gpu_bootstrap.warm_pool)"
      to: "holoscan.schedulers.EventBasedScheduler"
      via: "self.scheduler(EventBasedScheduler(self, worker_thread_number=5, name=\"concurrent\")) set before run()"
    - from: ".planning/scripts/phase2_gate.py"
      to: "subprocess app runs"
      via: "env pass-through of HOLOSCAN_CONCURRENT_FRAGMENTS / HOLOSCAN_GPU_RESAMPLE + new --json-out arg"

user_setup: []
---

<objective>
Ship the D-21 lever (concurrent independent fragments via `EventBasedScheduler`) and de-risk the
Phase 3 memory baseline by re-verifying the RMM initial-pool question (research Open Q1) before
any VRAM-sensitive work lands.

Purpose: Bottleneck #2 in `02-BENCHMARK-REPORT.md` §6 (serial fragment scheduling, up to ~25 s
potential; D-16 note: the per-config CudaStreamPools are wired but the default GreedyScheduler
runs all operators serially on one thread). The research live-probed that
`EventBasedScheduler(app, worker_thread_number=N)` is the 4.2-supported mechanism (GIL released
at the C++ boundary, CountCondition joins safe) and that the win is hiding CPU-bound scipy
resample/postresample spans behind other fragments' GPU inference — NOT overlapping two
GPU-saturated inferences.

Output: Scheduler swap behind a config flag, full gate suite re-run both ways, nsys overlap
evidence, and the Open-Q1 RMM verification/pin.
</objective>

<execution_context>
@~/.claude/get-shit-done/workflows/execute-plan.md
@~/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/ROADMAP.md
@.planning/phases/03-optimization/03-CONTEXT.md (D-20..D-26 — locked; D-21 and D-25 control this plan)
@.planning/phases/03-optimization/03-RESEARCH.md (§D-21 Domain — wiring + thread-safety audit; Pitfalls 2/5/6/7/8; Open Q1)
@.planning/phases/02-gpu-acceleration/02-BENCHMARK-REPORT.md (§6 item 2 — the bottleneck this plan attacks)

# Files under change (read the relevant regions before editing):
@examples/apps/cchmc-nnunet-fast/my_app/app.py (compose() at ~line 357; subgraph loop ~455–480; tail warm_pool ~line 615; `if __name__ == "__main__"` run)
@examples/apps/cchmc-nnunet-fast/my_app/gpu_bootstrap.py (rmm.reinitialize site, warm_pool)
@.planning/scripts/phase2_gate.py (run_subprocess / run_fast_app env handling; main() JSON output path)
@.planning/scripts/nsight_profile_phase2.sh (Phase 2 nsys capture pattern — reuse for .planning/profiles/phase3/)
@.planning/scripts/probes-phase3/ (committed provenance for the research's live probes)

<interfaces>
<!-- Extracted from the codebase — use directly, no exploration needed. -->

From examples/apps/cchmc-nnunet-fast/my_app/app.py:
```python
# compose() tail (after all flows):
gpu_bootstrap.warm_pool(plan.total_bytes)
logging.info(f"End {self.compose.__name__}")

# app entrypoint:
if __name__ == "__main__":
    CCHMCNNUnetFastApp().run()
```
Scheduler must be set at the end of `compose()` (after `warm_pool`, before the End log) — the
research verified `self.scheduler(...)` works at that point (compose runs during construction,
before `run()`).

From .planning/scripts/phase2_gate.py:
```python
def run_fast_app(model_list_env, out_dir: Path):  # builds env from os.environ, pops HOLOSCAN_MODEL_LIST
def pixel_diff(fast_dir: Path, oracle: str, json_path: Path):
def main(argv=None) -> int:   # writes combined JSON (02-GATE-RESULTS.json today)
```
`main()` currently hard-codes the output JSON path under the Phase 2 gates dir — add a
`--json-out` CLI arg (default = current path, so Phase 2 invocation is unchanged).

Research-verified scheduler API (holoscan-cu13 4.2.0, live-probed):
```python
from holoscan.schedulers import EventBasedScheduler
self.scheduler(EventBasedScheduler(self, worker_thread_number=5, name="concurrent"))
```
worker_thread_number=5 = 3 fragment chains + app-level postprocess + writers (each chain has at
most one ready operator at a time; headroom is free). Do NOT use MultiThreadScheduler (polling
CPU overhead vs CPU-bound scipy spans), make_thread_pool, or CudaGreenContextPool (research:
explicitly not recommended for Phase 3).
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Re-verify RMM initial-pool reservation (Open Q1) and pin if drifted</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/gpu_bootstrap.py, .planning/profiles/phase3/</files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/my_app/gpu_bootstrap.py (the rmm.reinitialize call — currently `rmm.reinitialize(pool_allocator=True, managed_memory=False)` with no initial_pool_size)
    - .planning/phases/03-optimization/03-RESEARCH.md — Open Question 1 (rmm 26.2.0 default initial pool = ½ total GPU memory = 20 GiB measured 2026-08-19 vs Phase 2 trace's ~1.98 GB total cudaMalloc) + Pitfall 8 (scratch venv drifts) + Pitfall 7 (pin CUDA_VISIBLE_DEVICES, record device)
    - .planning/profiles/phase2/ phase2_bundle_cuda_api_sum.txt (Phase 2 baseline: 9 cudaMalloc / 1 cudaFree)
    - AGENTS.md (venv /tmp/monai-env/.venv; commit early)
  </read_first>
  <action>
    1. In a fresh venv subprocess (RMM reinitialize happens at import — run from examples/apps/cchmc-nnunet-fast with `ulimit -s unlimited`), after `my_app.gpu_bootstrap` import: sample driver-level memory with pynvml (`nvmlDeviceGetMemoryInfo`) and record the immediate post-reinit reservation S on a known-free device (Pitfall 7).
    2. Run one bundle rep (HOLOSCAN_MODEL_LIST unset) under nsys with FULL-PROCESS capture (no `--capture-range=cudaProfilerApi` — the setup/RMM-warm span must be in the trace, Phase 2 Plan 06 convention), reusing `.planning/scripts/nsight_profile_phase2.sh`'s pattern; output to `.planning/profiles/phase3/`. Export `cuda_api_sum`. Check the bootstrap cudaMalloc size.
    3. Decision (record in `.planning/profiles/phase3/rmm_openq1.md`):
       - If S is small (Phase 2-like, ≲3 GiB): NO code change — record the finding (env did not drift for RMM purposes) + the re-run churn counts (cudaMalloc/cudaFree vs kernel launches) as the correct INFR-02 baseline for Plan 03.
       - If S ≥ 10 GiB (the 20 GiB reservation is real): pin it in `gpu_bootstrap.py` — change the reinitialize call to pass `initial_pool_size=4 * 1024**3` (4 GiB: covers the airway bundle budget total, which compose logs as `memory_budget: {"total_bytes": ...}` — read the value from a fresh run's log and cite it in a comment; the existing `warm_pool(plan.total_bytes)` at compose end still grows the pool per D-14, so the pin only removes the wasteful default reservation). Add a `self._logger`-style log line (module-level `logging.info("rmm initial_pool_size: ...")`) at the reinitialize site. Keep the import-order invariant untouched (rmm BEFORE any holoscan/torch CUDA init — Pitfall 2).
       - After any pin: re-run the single bundle rep under nsys and re-export cuda_api_sum to confirm the churn baseline (pool expansions only, never per-tile).
    4. `.planning/profiles/phase3/` must end with: the two nsys rep+sqlite pairs, cuda_api_sum exports, and `rmm_openq1.md` containing: measured S, the decision (pinned value or no-change), device id, and the before/after cudaMalloc counts.
    Commit after this task (short imperative message).
  </action>
  <verify>
    <automated>test -f .planning/profiles/phase3/rmm_openq1.md && test -d .planning/profiles/phase3 && ls .planning/profiles/phase3/*.sqlite | wc -l | grep -qE '^[2-9]'; grep -c "initial_pool_size" examples/apps/cchmc-nnunet-fast/my_app/gpu_bootstrap.py  # ≥1 iff pinned; 0 otherwise (record which in rmm_openq1.md)</automated>
  </verify>
  <acceptance_criteria>
    - `rmm_openq1.md` exists with the measured post-reinit reservation size (GiB), the decision with reasoning, the pinned `initial_pool_size` value (or explicit no-change), the CUDA device id, and before/after cudaMalloc counts.
    - If pinned: `grep -c "initial_pool_size" my_app/gpu_bootstrap.py` ≥ 1 and a fresh bundle rep still exits 0 with `memory_allocator_backend: pluggable` in the log.
    - If not pinned: the doc explicitly states S < 10 GiB with the number, and Plan 03's churn baseline cites the re-run cuda_api_sum.
  </acceptance_criteria>
  <done>The RMM baseline for Phase 3 is measured against the LIVE venv (not the 2026-08-19 research snapshot), the pool reservation is pinned or documented, and the churn baseline is re-established for Plan 03's INFR-02 proof.</done>
</task>

<task type="auto">
  <name>Task 2: D-21 scheduler swap + gate suite both ways + nsys overlap evidence</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/app.py, .planning/scripts/phase2_gate.py, .planning/profiles/phase3/, .planning/phases/03-optimization/gates/</files>
  <read_first>
    - examples/apps/cchmc-nnunet-fast/my_app/app.py — compose() in full (lines ~357–620), the subgraph loop, and the tail after gpu_bootstrap.warm_pool (the insertion point)
    - .planning/phases/03-optimization/03-RESEARCH.md — §D-21 Domain: the wiring snippet, the thread-safety audit table (timing record order nondeterminism, set_num_threads interleave, CuPy default-stream serialization), the "stream note: keep legacy default streams" anti-pattern list, Open Q3 (worker count sizing)
    - .planning/scripts/phase2_gate.py — run_subprocess/run_fast_app (env construction), main() (JSON output), the 4 gate rows
    - .planning/phases/02-gpu-acceleration/02-BENCHMARK-REPORT.md — §5 (Phase 2 single-stream back-to-back finding = the contrast baseline for the overlap note)
  </read_first>
  <action>
    1. **Gate script (backward-compatible extension):** in `.planning/scripts/phase2_gate.py` add a `--json-out PATH` argument to `main()` (default = the existing Phase 2 output path, so the Phase 2 invocation is unchanged) and ensure `run_fast_app`/`run_subprocess` forward `HOLOSCAN_CONCURRENT_FRAGMENTS` and `HOLOSCAN_GPU_RESAMPLE` from the parent environment to the app subprocesses (check the existing env construction; env is built from os.environ so pop/copy behavior must preserve these — verify explicitly).
    2. **Scheduler swap in app.py:** add `from holoscan.schedulers import EventBasedScheduler` to the imports, and at the END of `compose()` (after `gpu_bootstrap.warm_pool(plan.total_bytes)`, before the `End {compose}` log):
       ```python
       # D-21: concurrent independent-fragment execution. Default OFF = the
       # Phase 2 GreedyScheduler serial behavior (byte-for-byte unchanged).
       # Flips to default-ON in step 5 below only if the full gate suite
       # passes with concurrency enabled.
       if os.environ.get("HOLOSCAN_CONCURRENT_FRAGMENTS") == "1":
           self.scheduler(EventBasedScheduler(self, worker_thread_number=5, name="concurrent"))
           self._logger.info("scheduler: EventBasedScheduler worker_thread_number=5 (D-21)")
       else:
           self._logger.info("scheduler: default GreedyScheduler (serial, Phase 2 behavior)")
       ```
       Do NOT touch the per-config CudaStreamPools (INFR-004 as-is), do NOT introduce per-fragment non-blocking streams or green contexts (research anti-patterns — measured ≈0 GPU-overlap gain).
    3. **Gate suite, both ways (D-25 anchor; the pixel-exact suite is the correctness gate):** pin a free GPU (`export CUDA_VISIBLE_DEVICES=<id>`; record the device id in the JSON note/summary) and run:
       - OFF regression: `/tmp/monai-env/.venv/bin/python .planning/scripts/phase2_gate.py --json-out .planning/phases/03-optimization/gates/03-GATE-serial.json` (flag unset — must reproduce the Phase 2 pass: fullres 99.99986%/3 documented fp16↔fp32 boundary class, lowres/cascade/bundle 100.00000%/0, SR 0.0%, residency PASS).
       - ON: `HOLOSCAN_CONCURRENT_FRAGMENTS=1 /tmp/monai-env/.venv/bin/python .planning/scripts/phase2_gate.py --json-out .planning/phases/03-optimization/gates/03-GATE-concurrent.json` — all 4 pixel gates + SR ≤0.1% + residency must PASS.
       Expected concurrency hazards from the research thread-safety audit: timing record order in `study_timing_summary` becomes nondeterministic (sort by start_ns if you assert anything about it); the fullres 3-voxel boundary is the sensitive gate (set_num_threads interleaving) — if ONLY that gate flips by 1–3 voxels in the documented fp16↔fp32 boundary class, record it in the JSON note; anything else is a regression.
    4. **nsys overlap evidence:** one bundle rep with `HOLOSCAN_CONCURRENT_FRAGMENTS=1` under full-process nsys capture → `.planning/profiles/phase3/` (rep + sqlite + `nvtx_sum` + `nvtx_kern_sum` exports). Write `.planning/profiles/phase3/overlap.md`: query the sqlite NVTX ranges for `preprocess_3d_*` / `inference_3d_*` / `postresample_3d_*` and show the time-window overlap table (which spans of which configs overlap, with start/end timestamps) against the Phase 2 §5 contrast (single stream id 7, back-to-back, 0.2 ms gap). This file is the trace citation for D-21 (and the fallback evidence if step 5 takes the fallback).
    5. **Flag default decision (D-21 fallback clause):** if the ON gate run is fully green → flip the default: change the condition to `os.environ.get("HOLOSCAN_CONCURRENT_FRAGMENTS", "1") != "0"` (concurrency ON by default; explicit `=0` restores the serial fallback) and commit. If any gate fails → keep the default OFF, write the measured ceiling (what overlap the trace achieved, which gate regressed and by how much) into `overlap.md` under a "## Measured ceiling (fallback shipped)" heading, and commit.
    Commit after this task.
  </action>
  <verify>
    <automated>grep -q "EventBasedScheduler" examples/apps/cchmc-nnunet-fast/my_app/app.py && grep -q "HOLOSCAN_CONCURRENT_FRAGMENTS" examples/apps/cchmc-nnunet-fast/my_app/app.py && grep -q "json-out" .planning/scripts/phase2_gate.py && test -f .planning/phases/03-optimization/gates/03-GATE-concurrent.json && test -f .planning/phases/03-optimization/gates/03-GATE-serial.json && test -f .planning/profiles/phase3/overlap.md && /tmp/monai-env/.venv/bin/python -c "import json; d=json.load(open('.planning/phases/03-optimization/gates/03-GATE-concurrent.json')); assert all(g.get('ok') or g.get('status')=='PASS' for g in d['gates'] if isinstance(g, dict)), 'gate not green'; print('concurrent gates green')"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "EventBasedScheduler" my_app/app.py` ≥ 1 and the wiring matches the research snippet (worker_thread_number=5, set at compose tail, flag-gated).
    - Both gate JSONs exist; the serial one reproduces Phase 2 (fullres 99.99986%/3 boundary class, others 100.00000%/0, SR 0.0%, residency PASS); the concurrent one is fully green (or the fallback note exists with the trace citation and the flag default is OFF).
    - `overlap.md` contains the per-config NVTX overlap table with timestamps and names the worker threads (distinct tids) vs Phase 2's single thread.
    - `phase2_gate.py --json-out` is backward compatible: invoking without the flag still targets the Phase 2 JSON path.
    - The final flag default state (ON or OFF) is explicit in the commit message and `overlap.md`.
  </acceptance_criteria>
  <done>D-21 is shipped (concurrent-by-default behind a serial fallback flag, or the measured ceiling documented with the serial default per the fallback clause), the D-25 gate suite passes in the shipping configuration, and the nsys overlap citation exists in .planning/profiles/phase3/.</done>
</task>

</tasks>

<verification>
- Both gate JSONs parse as valid JSON and every gate row is green in the shipping configuration (concurrent if flipped, serial if fallback).
- `git log` shows the two commits for this plan (imperative messages).
- A fresh serial-mode bundle run (flag unset / `=0`) still exits 0 and logs `scheduler: default GreedyScheduler`.
- No changes outside the three listed files (plus artifacts under `.planning/profiles/phase3/` and `.planning/phases/03-optimization/gates/`).
</verification>

<success_criteria>
- D-21 outcome is binary and recorded: concurrent default ON with a fully green gate suite, OR serial default with the measured-ceiling note + trace citation (D-21 fallback clause).
- The RMM baseline for the phase is live-measured (Open Q1 resolved or pinned) and the churn baseline for Plan 03 is re-established.
- Phase 2 evidence is untouched (02-GATE-RESULTS.json, .planning/profiles/phase2/ unchanged).
</success_criteria>

<output>
After completion, create `.planning/phases/03-optimization/3-optimization-01-SUMMARY.md` covering: the flag default shipped (ON/OFF) with the gate numbers, the Open-Q1 finding + pin decision, the overlap table headline (which spans overlapped, measured wall delta if any single-config run is available), commits, and any deviations (Rule 1–3 style) from this plan.
</output>