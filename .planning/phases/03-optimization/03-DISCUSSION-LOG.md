# Phase 3: Optimization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md - this log preserves the alternatives considered.

**Date:** 2026-08-19
**Phase:** 03-optimization
**Mode:** auto (yolo auto-advance from Phase 2 completion; recommended options selected and logged for user review)
**Areas discussed:** Scope sizing, Bottleneck strategy, Memory lifecycle, INFR-02 proof, Gates & external dependencies

---

## Scope sizing

| Option | Description | Selected |
| ------ | ----------- | -------- |
| Trimmed data-driven sweep | Two low-correctness-risk measurable levers (fragment scheduling, memory lifecycle) + INFR-02 + gated GPU-resample experiment + final benchmark/validation | ✓ |
| Full sweep | + TensorRT export, torch.compile, MEM-01 weight sharing | |
| Minimal close | INFR-02 + final benchmark/validation only, no new optimization work | |

**Agent's choice (auto):** Trimmed data-driven sweep (D-20).
**Rationale:** Phase 2 met both D-18 bars (1.082× / 1.310×) and the ROADMAP explicitly makes Phase 3 conditional ("may be skipped or trimmed"). The deferred items' expected wins (inference-kernel level) are unvalidatable while ncu is admin-blocked, and TensorRT/torch.compile carry High/Medium risk against an unknown per-kernel profile.

## Bottleneck strategy — fragment scheduling

| Option | Description | Selected |
| ------ | ----------- | -------- |
| Concurrent independent fragments | DAG/executor-level overlap (fullres ∥ lowres) via existing per-config streams; no math change; pixel-gates + residency gate; fallback to serial with measured ceiling if holoscan 4.2 can't express it | ✓ |
| Keep serial | Document as a limitation | |

**Agent's choice (auto):** Concurrent fragments (D-21).
**Rationale:** Bottleneck #2 (~25 s, up to ~25 s of a 76.7 s inference block), zero math risk, streams already wired (D-16 says scheduling — not pools — is the blocker).

## Bottleneck strategy — GPU resampling

| Option | Description | Selected |
| ------ | ----------- | -------- |
| Gated experiment | CuPy/monai GPU resampler (image + seg) behind a config flag, default OFF; per-tensor np.array_equal vs scipy on all 4 configs; ship CPU default if not byte-identical, document divergence | ✓ |
| Defer beyond v2 | Skip ROADMAP 3.4 | |
| Full port | Commit to GPU resample, relax pixel-exact bar to documented tolerance | |

**Agent's choice (auto):** Gated experiment (D-22).
**Rationale:** Largest single lever (~28.8 s, 22.2% of bundle E2E). D-13 explicitly designated GPU resampling the v2 item and Phase 3 is that slot (ROADMAP 3.4: "If preprocessing is the bottleneck: evaluate GPU resampling"). The gated form keeps the phase pixel-exact by construction — the byte-identity gate is the arbiter (ROADMAP Critical-risk mitigation: "may need to keep CPU path").

## Memory lifecycle

| Option | Description | Selected |
| ------ | ----------- | -------- |
| MEM-003 only | Free lowres weights after inference + measured peak-VRAM delta; MEM-01 evaluation note; MEM-02 hardware-unverifiable | ✓ |
| MEM-003 + MEM-01 | Also shared weights across 15-ckpt ensemble | |
| All three | + 8 GB VRAM target | |

**Agent's choice (auto):** MEM-003 only (D-23).
**Rationale:** MEM-003 is a safe, measured memory-lifecycle change (peak-VRAM delta is directly measurable). MEM-01 has no profiling support (models load once; inference saturated). MEM-02 requires an 8 GB GPU — this machine is A100-SXM4-40GB, so it cannot be verified here.

## INFR-02 proof strategy

| Option | Description | Selected |
| ------ | ----------- | -------- |
| Layers: synthetic unit tests + multi-study replay (nvidia-smi / cudaMalloc-churn flat) + user reference examples when they arrive | Ship without the examples if they don't land; record as external-dependency item (D-24) | ✓ |
| Hard-block on user reference examples | Phase cannot complete until examples arrive | |

**Agent's choice (auto):** Layered proof with external-dependency fallback (D-24).
**Rationale:** D-17 parked INFR-02 here with the user adding reference examples — but "provable" must not mean "unfinishable"; the 2d-model precedent (D-01/D-03) is the established pattern for user-supplied dependencies.

## Gates & external dependencies

| Option | Description | Selected |
| ------ | ----------- | -------- |
| Dev-corpus close + 3 external-dependency items (≥5-CT corpus, ncu admin, INFR-02 examples) | Non-blocking; re-open as gap plans if dependencies land | ✓ |
| Hard-block phase completion | Wait for corpus + ncu | |
| Drop the ≥5-CT bar | Remove TEST-01 final gate | |

**Agent's choice (auto):** Dev-corpus close with recorded external-dependency items (D-26).
**Rationale:** Mirrors D-01/D-03; keeps the v1.0 milestone finishable while preserving every unmet bar as traceable debt.

## the agent's Discretion

- Concurrency mechanism for D-21 (live-probe holoscan 4.2 first)
- GPU resample algorithm choice for D-22 (byte-identity gate arbitrates)
- Buffer-cache granularity / eviction for D-24

## Deferred Ideas

TensorRT (ACCEL-01/02), torch.compile (ACCEL-03), MEM-01, MEM-02, inference-kernel tuning (ncu-blocked), ≥5-CT re-run, INFR-02 user examples, 2d model validation, bootstrap caching (usage-model dependent). See CONTEXT.md <deferred>.
