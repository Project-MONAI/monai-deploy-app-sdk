# Resume Instructions — cchmc-nnunet-fast

**Written:** 2026-08-15 · **Branch:** `nnunet-fast` · **Single source of truth:** `.planning/STATE.md` (see its "Resume after driver update" section — keep both in sync)

## Where we are

- **Phase 0 (Foundation), 5/7 tasks done.** App scaffolded, cu13 venv repaired and import-verified, Nsight/NVTX harness in place, RMM script ready.
- **Environment blockers:** all cleared except the **GPU driver** — host driver is 570.211.01 (CUDA 12.8), too old for the cu13 runtime (torch 2.13.0+cu130, cupy-cuda13x 13.6, rmm-cu13 26.2). Driver update (→ R580+/CUDA 13) is in progress.
- **Remaining Phase 0 acceptance work:** tasks 0.3 (reference corpus), 0.4 (baseline benchmark script), 0.5 (baseline results CSV), 0.6 (demo Nsight trace), 0.7 (RMM verification — auto-passes once the driver works).

## Step 1 — Verify the driver took effect

```bash
nvidia-smi | head -4                          # expect "CUDA Version: 13.x", R580+ driver
/tmp/monai-env/.venv/bin/python .planning/scripts/test_rmm.py
```

- Output shows `cudaAsync` / passes → driver blocker **cleared**. Update `.planning/STATE.md` (Blockers + acceptance row #5 → ✓) and commit.
- Still `SKIP` / `cudaErrorInsufficientDriver` → reboot is usually required after a Linux driver swap; check the driver package version too. **Do not continue past this gate.**

## Step 2 — Start a fresh pi session in the repo

```bash
cd /users/srv-mde/projects/monai-deploy-app-sdk && pi
```

Fresh context window is the GSD convention; it also clears any stale tool state from previous sessions.

## Step 3 — Run `/gsd-progress` first

It reads `.planning/STATE.md`, reports the position (Phase 0 of 4), and routes to the next action.

## Step 4 — Finish Phase 0 (in order)

| Task | What | Suggested route |
|------|------|-----------------|
| 0.3 | Reference corpus: ≥5 CT studies run through `cchmc_nnunet_fifteen_ckpt_app`, ground-truth DICOM-SEG + DICOM-SR saved and checksummed (current `testdata/` holds only 1 MR series — not sufficient) | `/gsd-quick` |
| 0.4 | Baseline benchmark script — runs the current app on the corpus, records end-to-end + per-stage latency (CSV: study, total ms, per-stage ms) | `/gsd-quick` |
| 0.5 | Run it → `.planning/baseline_results.csv` (3 repetitions, mean ± std, warm-up first) | same session |
| 0.6 | Generate one demo Nsight trace via `.planning/scripts/nsight_profile.sh` (`.nsys-rep`/`.sqlite`/`.qdstrm`) to prove the harness end-to-end | same session |

Alternative: formalize instead of ad-hoc — `/gsd-plan-phase 0` → `/gsd-execute-phase 0` (project GSD config is YOLO/coarse, so planning is lightweight). Both paths end at the same gate.

## Step 5 — Phase gate, then Phase 1

1. All 5 Phase 0 acceptance criteria ✓ (see `.planning/ROADMAP.md`).
2. Update ROADMAP traceability matrix + PROJECT.md (move "Phase 0 completion" to Validated). Commit.
3. `/gsd-transition` — mark Phase 0 complete.
4. `/gsd-discuss-phase 1` — config is `discuss_mode=discuss` and Phase 1 has no CONTEXT.md yet; this surfaces the key decisions (operator boundaries, TTA flip order, RMM pool ownership).
5. `/gsd-plan-phase 1` → `/gsd-execute-phase 1` — Core Pipeline (Preprocess → SlideWindow → PostResample → Ensemble → Postprocess → DICOM-SEG, pixel-exact vs reference).

## Environment reference (already set up — don't redo)

- Venv: `/tmp/monai-env/.venv` (uv-managed). Activate: `source activate-env.sh`. Never use system Python; never pip-install `nnunetv2` (vendored editable at `./nnUNet/`).
- Packages (verified 2026-08-14/15): holoscan-cu13 4.2.0, holoscan-cli 4.2.0, cupy-cuda13x 13.6.0, rmm-cu13 26.2.0, torch 2.13.0+cu130, monai 1.3.0, pydicom 3.0.2, highdicom 0.28.1, nnunetv2 2.8.1 (editable), monai-deploy-app-sdk (editable).
- If `import holoscan.flow_graphs` ever fails with `libgxf_*.so: cannot open shared object file` → the wheel install is corrupted; repair per `examples/apps/cchmc-nnunet-fast/README.md` (force-reinstall holoscan-cu13, RECORD-audit the result).
- Static analysis (pyright/Pylance) points at the venv via `pyproject.toml [tool.pyright]` + `.vscode/settings.json`; `monai.deploy` import "errors" in editors are a known PEP 660-editable limitation, suppressed by config.
- Holoscan 4.x wants a 32 MB thread stack: `ulimit -s 32768` (or `--ulimit stack=33554432` in Docker).

## If the driver update doesn't stick (Plan B)

Use a CUDA 13 container (NVIDIA container toolkit + Holoscan container image). Consequences to carry into the session: GPU verification runs inside the container, and the `venvPath` values in `pyproject.toml` / `.vscode/settings.json` may need to change. Tell the new session which route was taken.
