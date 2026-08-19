# Phase 2: GPU Acceleration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md - this log preserves the alternatives considered.

**Date:** 2026-08-18
**Phase:** 2-GPU Acceleration
**Areas discussed:** 2D config coverage; Reference oracle & gate strategy; CuPy port fidelity; Infrastructure strictness

---

## 2D config coverage

| Option | Description | Selected |
| --- | --- | --- |
| 1a | Gate on the three 3D configs; fragment wiring stays config-generic; 2D documented as blocked-on-model (met-with-deviation, Phase 0 pattern) | ✓ |
| 1b | Block the phase gate until a real 2d model is obtained | |

**User's choice:** 1a
**Notes:** "using a dummy 2d model will require changes to the inference and ensemble configuration files which is not recommended. I will find a 2d model to test after all phases are complete and the airway model is fully tested."

---

## Reference oracle & gate strategy

| Option | Description | Selected |
| --- | --- | --- |
| 2a | Fresh per-config references (`ref_lowres_only`, `ref_cascade_only`) via the reference_fullres_run.py harness pattern for task 2.6; final gate = fast bundle ensemble (fullres + cascade_fullres) vs `testdata/current_output`; lowres standalone, not in bundle | ✓ |
| 2b | Single gate on `current_output` only; per-config outputs checked by structural smoke | |

**User's choice:** 2a — "no further comments."

### Sub-question: cascade one-hot handoff

| Option | Description | Selected |
| --- | --- | --- |
| 2i-a | lowres post-softmax argmax → one-hot float stack on GPU → extra channels into cascade_fullres preprocess (mirrors nnUNet cascade plans, zero disk I/O) | ✓ |
| 2i-b | feed raw lowres probabilities instead of argmax (diverges from nnUNet reference) | |

**User's choice:** 2i-a ("do a")

---

## CuPy port fidelity

| Option | Description | Selected |
| --- | --- | --- |
| 3a | Byte-identity per ported op vs numpy path (fp32, C-contiguous) + final SEG pixel diff per config (agent recommendation) | |
| 3b | Only the final pixel-exact gate; internal divergence found only if the gate fails | ✓ |

**User's choice:** 3b — "3b is fine."
**Notes:** Agent's fp32/C-contiguous hygiene (D-12) and the accepted GPU↔CPU round-trip around the scipy resample (D-13) remain in force regardless of the check strategy.

---

## Infrastructure strictness

| Option | Description | Selected |
| --- | --- | --- |
| 4a | RMM pre-allocation + budget calculator unit-tested with synthetic sizes (real OOM documented as unexercised); CudaStreamPool wired, overlap best-effort; latency bar = any positive E2E improvement vs 61.8 s with per-operator deltas | ✓ |
| 4b | Strict: visible stream overlap in traces required + hard ≥10% E2E cut | |

**User's choice:** 4a

### Sub-question: INFR-02 cross-study buffer reuse

| Option | Description | Selected |
| --- | --- | --- |
| 4i-a | Prove reuse by unit test (same buffer across two compute calls on synthetic inputs) + code assertion | |
| 4i-b | Defer INFR-02 to Phase 3 as unprovable on 1 study | ✓ |

**User's choice:** 4i-b — "I will add additional reference examples for phase 3." Phase 3 planning should expect a multi-study corpus to exist.

---

## Agent's Discretion

- Exact CuPy kernel choices / op fusion inside preprocess
- Multi-fragment DAG config exposure (file shape, fragment naming)
- Synthetic sizes for budget-calculator unit tests
- Structured timing log extensions
- INF-009 handling within the pixel-exact precedence rule (D-19)

## Deferred Ideas

- Real 2d model E2E validation — after all phases (user-supplied model)
- INFR-02 cross-study buffer-reuse proof — Phase 3 (user adding reference examples)
- ≥5-study final gate re-run — pre-existing, corpus pending
- GPU resampling — v2
