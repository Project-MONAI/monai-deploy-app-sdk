---
phase: "1-core-pipeline"
plan: "03"
type: "execute"
wave: 2
depends_on: ["01"]
files_modified:
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py"
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py"
  - "examples/apps/cchmc-nnunet-fast/my_app/operators/postprocess_operator.py"
autonomous: true
user_setup:
  - "jsonpkls/postprocessing.pkl present (CC size-threshold rules) for the airway model."
must_haves:
  truths:
    - "PostResampleOperator applies softmax -> argmax, resamples probabilities to original shape, and reverts crop + transpose to original DICOM orientation"
    - "EnsembleAverageOperator computes an in-memory element-wise mean of per-config probability maps and applies argmax AFTER averaging (no .npz disk I/O)"
    - "PostprocessOperator applies connected-component analysis using the CuPy / skimage.measure GPU path with the pkl size-threshold rules and the keep-largest-component step"
    - "PostprocessOperator produces output as a GPU tensor and transfers to CPU numpy exactly once at the pipeline boundary before the DICOM-SEG writer"
  artifacts:
    - "examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py"
    - "examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py"
    - "examples/apps/cchmc-nnunet-fast/my_app/operators/postprocess_operator.py"
  key_links:
    - "SlideWindowOperator logits -> PostResampleOperator -> per-config probabilities"
    - "per-config probabilities -> EnsembleAverageOperator -> averaged probabilities -> argmax seg"
    - "seg -> PostprocessOperator -> final uint8 seg (CPU, once) -> DICOMSegmentationWriterOperator"
---

# Phase 1 Plan 03: PostResample + EnsembleAverage + Postprocess Operators

## Objective
- **What:** Build the three post-inference operators: `PostResampleOperator`,
  `EnsembleAverageOperator`, `PostprocessOperator`.
- **Why:** These map patch logits back to original orientation, average the (currently single,
  future multi-) config probability maps **in GPU memory with no disk I/O** (the reference round-trips
  `.npz` through a spawn pool), and run connected-component cleanup — the tail of the chain.
- **Output:** Three operators that, fed real logits, reproduce the reference post-inference
  segmentation exactly.

## Context
@.planning/PROJECT.md
@.planning/STATE.md

Reference: `export_prediction`/`resample_and_save` does `resampling_fn_probabilities` →
softmax (`apply_inference_nonlin`) → revert-crop (`insert_crop_into_image`) → revert-transpose,
writing probabilities to `.npz`. `EnsembleProbabilitiesToSegmentation` loads each `.npz` and calls
`nnunetv2.ensembling.ensemble.average_probabilities` (float32 mean) → `convert_logits_to_segmentation`.
`PostProcessNNUnet` applies `postprocessing.pkl` rules; `KeepLargestConnectedComponentd(applied_labels=[1])`.
All currently CPU + disk. Phase 1 keeps the **math** identical but moves it to GPU and removes the
`.npz` round-trip.

Scope: Phase 1 = single config `3D_fullres`, so the ensemble mean is over one config — but the
operator MUST accept a list of per-config probability tensors (in-memory) so Phase 2 adds configs
and cascade without redesign.

## Tasks

<task type="code">
  <name>1.9 PostResampleOperator</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/postresample_operator.py</files>
  <action>Implement a Holoscan operator subclass. Input: logits (1, n_classes, x',y',z') GPU + metadata from Preprocess (bbox, pre-crop shape, spacing, transpose). Compute: softmax over classes, resample probability volume to the original (pre-crop) shape using the reference resampling path, revert crop (insert into full volume) and revert transpose to original DICOM orientation. Output: per-config probability volume (n_classes, z,y,x) GPU. Keep assert_on_gpu at boundaries.</action>
  <verify>On 3D_fullres airway logits, the reverted probability volume matches the reference `resample_and_save` output (shape + values) before argmax.</verify>
  <done>Segmentation mapped back to original DICOM orientation and shape (POST-02).</done>
</task>

<task type="code">
  <name>1.11 EnsembleAverageOperator (in-memory, no disk)</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/ensemble_average_operator.py</files>
  <action>Implement a Holoscan operator subclass that accepts a list/tensor stack of per-config post-softmax probability maps and computes an in-memory element-wise mean in GPU memory (incremental in-place where possible). Apply argmax AFTER averaging (not before), matching nnUNet average_probabilities. No .npz I/O. Output: averaged probabilities (n_classes, z,y,x) GPU.</action>
  <verify>With a single 3D_fullres probability volume, output == input (mean of one); with a synthetic 2-config stack, output == element-wise mean, then argmax matches reference. No temp files created.</verify>
  <done>In-memory GPU probability averaging, argmax-after-average, zero disk I/O (INF-009, INF-010).</done>
</task>

<task type="code">
  <name>1.12 PostprocessOperator (CC on GPU)</name>
  <files>examples/apps/cchmc-nnunet-fast/my_app/operators/postprocess_operator.py</files>
  <action>Implement a Holoscan operator subclass. Input: averaged-probability seg (uint8, GPU). Compute: connected-component analysis using the CuPy / skimage.measure GPU path with the size-threshold rules from jsonpkls/postprocessing.pkl, plus keep-largest-component for applied_labels=[1]. Produce the final seg as a GPU tensor and transfer to CPU numpy EXACTLY ONCE at the pipeline boundary. Output: final uint8 seg (CPU) ready for DICOM-SEG writer; also emit the airway volume text for SR.</action>
  <verify>On the airway seg, the postprocessed output matches the reference PostProcessNNUnet + KeepLargestConnectedComponentd result (identical label mask). Exactly one GPU->CPU transfer occurs (verified by GPU-residency scan in plan 05).</verify>
  <done>CC cleanup on GPU, single CPU transfer at the boundary (POST-01, POST-03).</done>
</task>

## Verification
- All three operators exist as Holoscan operator subclasses with setup() and compute().
- PostResample reverts to original orientation; EnsembleAverage averages in-memory (no disk) and
  argmax-after-average; Postprocess runs CC on GPU with a single boundary CPU transfer.
- On the 3D_fullres airway logits, the combined post-inference seg matches the reference.

## Success Criteria
- [ ] PostResampleOperator: softmax, argmax-ready, resample to original shape, revert crop/transpose (POST-02)
- [ ] EnsembleAverageOperator: in-memory element-wise mean, argmax after average, no .npz I/O (INF-009, INF-010)
- [ ] PostprocessOperator: connected-component cleanup via CuPy/skimage GPU path + keep-largest (POST-01)
- [ ] PostprocessOperator: GPU tensor → single CPU transfer at the boundary (POST-03)
- [ ] Device asserted at each operator boundary; no silent CPU fallback (INF-005)
