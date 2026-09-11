---
name: agent_output_quality_v1
description: Deterministic second-pass verifier for monai-agent-lung-nodule evidence packs. Grades schema validity, physical plausibility, and reproducibility. Does not re-run inference.
license: Apache-2.0
allowed-tools: Bash
metadata:
  author: Khaled Younis
  tags:
    - verifier
    - MONAI Deploy
    - agentic
---

# Verifier: Agent Output Quality (v1)

## Purpose
Grades a `result_json` evidence pack emitted by `monai-agent-lung-nodule` (or any skill declaring the same
finding schema) against a fixed set of engineering invariants. This is a meta-skill in the same sense as
`skill_completeness_v1` in NVIDIA's medical-AI-skills convention: it does not perform inference and does
not judge clinical correctness — it judges whether the primary skill's output is trustworthy *as evidence*.

## Gates
| Gate | Check |
|---|---|
| `schema` | `result_json` parses as JSON and contains `study_id`, `status`, `nodules_detected`, `findings`, `execution_metadata`. |
| `sanity` | `nodules_detected == len(findings)`; every finding has `AI_Generated: true` and a non-null `Verification_Status`. |
| `physical_bounds` | Every finding's `max_diameter_mm` in `[0, 100]`; `volume_mm3` (or `volume_voxels`) is non-negative. |
| `runtime` | `execution_metadata.inference_time_ms` present and within the bound declared by the primary skill's manifest. |
| `reproducibility` | Two invocations of the primary skill in `--mock` mode against the same `study_id` produce byte-identical `findings`. |

## Instructions
- Run `scripts/verify_result.py PATH_TO_result_json.json` against a completed skill run.
- To also check reproducibility, pass `--rerun-cmd "python ../../scripts/run_agent_lung_nodule.py --mock --study-id ..."` so the verifier can invoke a second run itself and diff it.
- Exit code 0 = all gates passed. Non-zero = at least one gate failed; stdout lists which.

## Available Scripts
| Script | Purpose | Arguments |
|---|---|---|
| `scripts/verify_result.py` | Runs all five gates against a `result_json` file. | `RESULT_JSON_PATH [--rerun-cmd CMD]` |

## Limitations
- Does not compute Dice/IoU or assess clinical accuracy — this skill audits engineering invariants only,
  matching the "verifier does not re-run inference / grades evidence, not clinical correctness" convention.
- The `reproducibility` gate only applies in `--mock` mode; real mode uses a random synthetic tensor per
  run by design (architecture validation, not determinism).
