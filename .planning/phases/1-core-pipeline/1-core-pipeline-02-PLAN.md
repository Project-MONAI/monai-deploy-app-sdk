---
phase: "1-core-pipeline"
plan: "02"
type: "execute"
wave: 2
depends_on: ["01"]
files_modified:
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py"
autonomous: true
user_setup:
  - "Models present at examples/apps/cchmc_nnunet_fifteen_ckpt_app/models (3d_fullres available)."
must_haves:
  truths:
    - "SlideWindowOperator loads model weights + architecture in setup()/on_insert(), not in compute() (no per-study cold start)"
    - "Runs MONAI sliding_window_inference with the same patch size, overlap (tile_step_size=0.5), and Gaussian weighting as the reference nnUNet predictor"
    - "Applies TTA mirror flips in the exact nnUNet order (normal + all mirror permutations from allowed_mirroring_axes)"
    - "TTA results accumulated with FP32 accumulators preserving sequential += order"
    - "Runs on GPU in eager mode (torch.no_grad) without CPU fallback; device asserted at the boundary"
    - "autocast scope is at the outermost inference boundary only (FP16 not split across operators)"
    - "Config-driven: patch size, overlap, tiling, checkpoint path all come from config; no hard-coded trainer class"
  artifacts:
    - "examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py"
  key_links:
    - "PreprocessOperator MemoryData(GPU) -> SlideWindowOperator -> logits tensor"
    - "nnUNet checkpoint (jsonpkls + 3d_fullres/*.pt) -> model built in setup()"
---

# Phase 1 Plan 02: SlideWindowOperator (Inference Core)

## Objective
- **What:** Build `SlideWindowOperator` — loads the nnUNet model once in `setup()`, then runs
  tile-based sliding-window inference with TTA in `compute()`, keeping everything on GPU.
- **Why:** This is the inference heart (≈138 s of the 169.7 s baseline) and the hardest operator
  to get numerically identical. Getting setup-vs-compute loading and TTA/accumulation order right
  here is the difference between pixel-exact and silently-wrong.
- **Output:** A GPU inference operator producing per-config logits, config-driven, with the
  numeric-equivalence guarantees (FP32 accumulator, TTA order, autocast scope).

## Context
@.planning/PROJECT.md
@.planning/STATE.md

Reference: `get_nnunet_monai_predictor` builds `nnUNetPredictor(tile_step_size=0.5,
use_gaussian=True, use_mirroring=True, device=cuda)`; `predict_sliding_window_return_logits`
pads to patch size, `compute_steps_for_sliding_window(..., tile_step_size=0.5)`, wraps in
`torch.autocast(device, enabled=True)`, gaussian smoothing, TTA mirrors (order = normal + all
mirror permutations of `allowed_mirroring_axes`), per-fold average. Checkpoint auto-order:
`final_model.pt`, `best_model.pt`, `model.pt`. The reference re-creates the whole ensemble in
every `compute()` (cold start) and downcasts input to CPU — both are the pitfalls we must NOT copy.

Scope: single config `3D_fullres`; operator is config-driven for Phase 2. TTA order and FP32
accumulator MUST match nnUNet exactly.

## Tasks

<task type="code">
  <name>1.3 SlideWindowOperator (model load in setup)</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py</files>
  <action>Implement a Holoscan operator subclass. setup(spec): spec.input("preprocessed"), spec.output("logits"). on_insert()/setup(): load the model architecture + weights for the configured config from the bundle checkpoint path (final_model.pt > best_model.pt > model.pt), .to('cuda').eval(); store on operator state. compute(): torch.no_grad(); run MONAI sliding_window_inference over the incoming GPU tensor with the config's patch_size, overlap_ratio (=tile_step_size 0.5), and gaussian_mode=True; return logits tensor on CUDA. NO model load in compute().</action>
  <verify>With a fresh process, model weights are loaded exactly once during setup (verify by timing or a load-counter log); a second study's compute() shows no model-load cost. Output logits shape matches the reference predictor for 3D_fullres.</verify>
  <done>Model loaded in setup; compute() does inference only; no cold start per study.</done>
</task>

<task type="code">
  <name>1.4 + 1.5 TTA mirrors in nnUNet order with FP32 accumulation</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py</files>
  <action>Replicate nnUNet TTA: build the mirror-permutation list from allowed_mirroring_axes in the exact reference order (normal first, then all allowed mirror permutations). Accumulate results into an FP32 accumulator using sequential += (same order). Do NOT use FP16 accumulation (non-associative). Keep autocast confined to the innermost model forward; do the TTA summing in FP32.</action>
  <verify>For a fixed airway input, the accumulated logits match the reference nnUNet predictor's TTA output (max abs diff within float tolerance ~1e-6, and identical after argmax/softmax to the reference seg).</verify>
  <done>TTA order + FP32 sequential accumulation reproduce the reference numerics.</done>
</task>

<task type="code">
  <name>1.6b + 1.10 Device assertion + autocast scope</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py</files>
  <action>Use gpu_util.assert_on_gpu at compute() entry (incoming preprocessed) and exit (outgoing logits). Wrap autocast at the outermost inference boundary only; assert no silent CPU fallback (raise on RuntimeError/OOM rather than swallowing). If nnUNet's OOM handler is in the path, wrap it to re-raise.</action>
  <verify>A forced-CPU input raises at entry; the OOM catch does not silently return a CPU tensor.</verify>
  <done>Device invariant enforced; autocast not split across operator boundaries.</done>
</task>

<task type="code">
  <name>1.7 + 1.8 Config-driven + custom trainer via checkpoint</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/slidewindow_operator.py, examples/apps/cchmc-nnunet-fast/my_app/config/__init__.py</files>
  <action>Read patch_size, overlap_ratio, gaussian flag, mirror axes, and checkpoint path from config (plans.json + bundle layout) keyed by config_name. Build the network from the checkpoint (build_network_architecture from the plans) rather than a hard-coded trainer class, so custom trainer variants load via checkpoint path.</action>
  <verify>Instantiating with config_name='3d_fullres' loads the correct checkpoint and patch size; pointing at a different checkpoint path loads that trainer's weights.</verify>
  <done>Inference is config- and checkpoint-driven, no hard-coded trainer assumptions.</done>
</task>

## Verification
- Fresh-process run loads the model once in setup; compute() is inference-only.
- Logits for 3D_fullres on the airway study match the reference predictor (TTA order + FP32 accumulator verified).
- Device asserted at both boundaries; no CPU fallback; autocast scoped to the inference boundary.

## Success Criteria
- [ ] Model loaded in setup()/on_insert(), never in compute() (INF-008)
- [ ] Runs on GPU in eager mode without CPU fallback (INF-001)
- [ ] MONAI sliding_window_inference with reference patch size, overlap, Gaussian (INF-002)
- [ ] TTA mirror flips in exact nnUNet order (INF-003)
- [ ] FP32 accumulator, sequential += order (INF-004)
- [ ] assert_on_gpu at boundaries, never swallows RuntimeError (INF-005)
- [ ] Config-driven patch/overlap/tiling (INF-006); custom trainer via checkpoint path (INF-007)
- [ ] autocast at outermost inference boundary only (INF-011)
