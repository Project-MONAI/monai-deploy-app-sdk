---
name: monai-agent-lung-nodule
description: Agent-callable wrapper exposing a MONAI Deploy lung-nodule segmentation operator as an MCP-style tool, with structured, schema-validated output for downstream clinical review. Not for clinical interpretation.
license: Apache-2.0
allowed-tools: Bash
metadata:
  author: Khaled Younis
  tags:
    - MONAI Deploy
    - agentic
    - MCP
    - thoracic
    - segmentation
---

# MONAI Agent: Lung Nodule Segmentation Tool

## Purpose
- Exposes a MONAI Deploy lung-nodule segmentation operator (`SegResNet`-based) as a single agent-callable
  tool, so an LLM or agent orchestrator can request "screen this CT for pulmonary nodules" without
  reading operator source, notebooks, or manuals.
- Demonstrates the general pattern proposed for MONAI Deploy: any existing `Operator` can be wrapped this
  way and registered in an MCP tool registry, discoverable and invocable by an agent at runtime instead of
  being hardcoded into a fixed execution graph.
- Manifest I/O: inputs are `dicom_study_id` (or a local NIfTI/CT volume path) and an optional
  `sensitivity_threshold`; outputs are a `result_json` evidence pack containing per-nodule findings and a
  Fleischner Society follow-up recommendation per finding.
- Not for clinical interpretation, autonomous diagnosis, or regulatory submission. Every output is marked
  `AI_Generated: true` and requires clinician sign-off before any clinical use.

## Instructions
- Read `skill_manifest.yaml` before changing arguments, side effects, or validation gates.
- Run `scripts/run_agent_lung_nodule.py` through the documented command below.
- If a host agent exposes `run_script`, use
  `run_script("scripts/run_agent_lung_nodule.py", args=[...])`; otherwise run the Python command shown below.
- Real inference requires `torch` and `monai` installed; when either is unavailable the script falls back
  to a deterministic simulation mode so the tool contract and downstream verifier can still be exercised
  without a GPU.
- Check the emitted `result_json` and the paired verifier's report before treating a run as evidence.

## Available Scripts
| Script | Purpose | Arguments |
|---|---|---|
| `scripts/run_agent_lung_nodule.py` | Primary entrypoint declared by `skill_manifest.yaml`. | `--study-id ID [--sensitivity-threshold 0.0-1.0] [--output-dir OUT_DIR] [--mock]` |

## Prerequisites
- Runtime requirements: Python 3.10+. Real mode additionally requires `torch` and `monai` (GPU/CUDA used
  automatically when available; CPU fallback supported but slower).
- Side effects: writes `result_json` under a caller-provided `--output-dir`; downloads no external data —
  the current implementation operates on a synthetic 3D tensor for architecture validation. A production
  version would source the volume from `dicom_series_to_volume_operator` per MONAI Deploy convention.
- Run commands from the skill directory root unless otherwise noted.

## Limitations
- This is a reference implementation for the agentic-wrapper pattern, not a validated clinical model. The
  segmentation backbone is an untrained `SegResNet` instance; findings are architecturally real (real
  forward pass, real tensor shapes, real timing) but not clinically meaningful until wired to a trained,
  validated bundle (e.g. via `monai_bundle_inference_operator`, matching `ai_spleen_seg_app` convention).
- No PHI is read, stored, or transmitted by this skill. It operates on a synthetic volume or a caller-
  supplied de-identified/public volume only (e.g. TCIA). See "Data governance" below.
- Device auto-detected (`cuda` if available, else `cpu`).
- Not for clinical deployment, clinical interpretation, autonomous diagnosis, or regulatory submission.

## Data governance
This skill changes how an already-permitted operator is *invoked* — it does not change where inference
runs or where data goes. Two supported deployment modes:
1. **Local / on-prem** — the MCP server and GPU inference run inside the institution's network, exactly as
   any MONAI Deploy MAP does today; nothing crosses the boundary that wasn't already crossing it.
2. **Public-dataset / research** — a hosted server operating on de-identified public data (e.g. TCIA); no
   PHI in scope.

A real clinical PHI use case (a clinician sending live patient data through a hosted MCP server) is a
separate, harder problem requiring data-use agreements and a security review — this skill does not attempt
to solve that, and should not be represented as solving it.

## Troubleshooting
| Error | Cause | Fix |
|---|---|---|
| `ImportError: No module named 'monai'` | Real mode requested without MONAI/PyTorch installed. | Install `monai` and `torch`, or omit `--mock`-forcing flags to use the auto-detected simulation fallback. |
| Empty or schema-invalid `result_json` | Wrong `--study-id`, or upstream operator failure. | Re-run with `--mock` against a known fixture and inspect stderr. |
| Validation gate failure | Output violated a declared invariant (e.g. missing `AI_Generated` flag, diameter/volume out of physical bounds). | Keep the failed evidence pack and use the paired verifier's gate message to repair inputs or wrapper code. |

## Exact Runnable Surface
```bash
python skills/monai-agent-lung-nodule/scripts/run_agent_lung_nodule.py \
  --study-id STUDY_CT_2026_0813 --sensitivity-threshold 0.80 --output-dir OUT_DIR
```
Add `--mock` to force the simulation path (no GPU/MONAI required) — useful for CI, where GPU is not
currently available (see `tests/`, which skip the real-mode assertions accordingly).

## Usage
```bash
python -m pip install torch monai   # optional — real mode only
python skills/monai-agent-lung-nodule/scripts/run_agent_lung_nodule.py \
  --study-id STUDY_CT_2026_0813 --output-dir vista_lung_outputs
```
The findings' Fleischner Society follow-up category and per-class volume/diameter checks are audited by
`verifiers/agent_output_quality_v1`, which does not re-run inference — it grades the emitted evidence pack
for schema validity, physical plausibility, and reproducibility.
