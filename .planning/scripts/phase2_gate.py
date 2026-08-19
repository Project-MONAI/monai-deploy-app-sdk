#!/usr/bin/env python3
"""phase2_gate.py — Phase 2 pixel-exact gate runner (TEST-01 / TEST-005).

Runs the four Phase 2 gate configurations of the fast app
(``cchmc-nnunet-fast``) and compares each against its per-config reference
oracle (D-05) / the reference full-bundle output (D-06), then runs the
GPU-residency static + runtime checks, and writes a combined machine-readable
report to ``.planning/phases/02-gpu-acceleration/gates/02-GATE-RESULTS.json``.

Gate matrix (fast-app side via ``HOLOSCAN_MODEL_LIST`` -> reference side):

  | gate          | HOLOSCAN_MODEL_LIST   | oracle                        |
  |---------------|-----------------------|-------------------------------|
  | fullres_only  | 3d_fullres            | testdata/ref_fullres_only     |
  | lowres_only   | 3d_lowres             | testdata/ref_lowres_only      |
  | cascade_only  | 3d_cascade_fullres    | testdata/ref_cascade_only     |
  | bundle        | (unset = default)     | testdata/current_output       |

Per gate row:
  1. fast app E2E run (venv python, subprocess, raised stack limit) into a
     scratch dir; require exit 0.
  2. ``pixel_diff.py <fast>/SEG <oracle> --json`` (defaults: 99.9% byte
     identity / 10000 differing voxels — the D-08 controlling
     segmentation-level tolerances; Phase 1 precedent ~99.999%+); require
     exit 0.
  3. SR airway-volume compare: parse "Airway Volume: N mL" out of both SR
     outputs; require relative delta <= 0.1%.
  4. assert the fast run's logged ``run_model_list=`` /
     ``ensemble_model_list=`` match the expected pair for the row — a wrong
     list would make the comparison meaningless.

After the 4 rows:
  5. ``gpu_residency.py --static`` and ``--runtime`` (runtime runs the app
     with HOLOSCAN_MODEL_LIST unset = the multi-fragment BUNDLE
     configuration); require exit 0. ``preprocess_operator.py`` stays in the
     deliberate D-13 allow-list (Plan 01); ``postprocess_operator.py``
     remains the only exactly-once final boundary.
  6. write the combined JSON report; print a summary table; exit non-zero
     iff any gate fails.

Usage (from anywhere; venv python recommended):
  /tmp/monai-env/.venv/bin/python .planning/scripts/phase2_gate.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PY = sys.executable

FAST_APP_ROOT = REPO_ROOT / "examples" / "apps" / "cchmc-nnunet-fast"
INPUT_DIR = REPO_ROOT / "testdata" / "airway_input"
MODEL_DIR = REPO_ROOT / "examples" / "apps" / "cchmc_nnunet_fifteen_ckpt_app" / "models"
PIXEL_DIFF = FAST_APP_ROOT / "scripts" / "pixel_diff.py"
GPU_RESIDENCY = FAST_APP_ROOT / "scripts" / "gpu_residency.py"
SCRATCH = Path("/tmp/phase2_gate")
REPORT = (REPO_ROOT / ".planning" / "phases" / "02-gpu-acceleration" / "gates"
          / "02-GATE-RESULTS.json")

# Gate rows: (row, HOLOSCAN_MODEL_LIST or None for the default bundle,
# reference oracle dir, expected run_model_list, expected ensemble_model_list)
# Expected lists per the Phase 2 Plan 04 gate table (resolve_run_model_list
# semantics: cascade auto-inserts lowres; ensemble excludes lowres; the
# fast app's documented self-ensemble fallback makes lowres-only runnable).
GATE_ROWS = [
    {
        "row": "fullres_only",
        "model_list_env": "3d_fullres",
        "oracle": str(REPO_ROOT / "testdata" / "ref_fullres_only"),
        "expected_run": ["3d_fullres"],
        "expected_ensemble": ["3d_fullres"],
    },
    {
        "row": "lowres_only",
        "model_list_env": "3d_lowres",
        "oracle": str(REPO_ROOT / "testdata" / "ref_lowres_only"),
        "expected_run": ["3d_lowres"],
        "expected_ensemble": ["3d_lowres"],
    },
    {
        "row": "cascade_only",
        "model_list_env": "3d_cascade_fullres",
        "oracle": str(REPO_ROOT / "testdata" / "ref_cascade_only"),
        "expected_run": ["3d_lowres", "3d_cascade_fullres"],
        "expected_ensemble": ["3d_cascade_fullres"],
    },
    {
        "row": "bundle",
        "model_list_env": None,  # unset = reference default (D-06)
        "oracle": str(REPO_ROOT / "testdata" / "current_output"),
        "expected_run": ["3d_fullres", "3d_lowres", "3d_cascade_fullres"],
        "expected_ensemble": ["3d_fullres", "3d_cascade_fullres"],
    },
]

LIST_RE = re.compile(
    r"run_model_list=(\[[^\]]*\])\s+ensemble_model_list=(\[[^\]]*\])")
SR_VOLUME_RE = re.compile(rb"Airway Volume:\s*([0-9.eE+-]+)\s*mL")


def run_subprocess(cmd, cwd, env, log_path):
    """Run a command, tee output to a log file, return (rc, stdout)."""
    with open(log_path, "w") as log:
        proc = subprocess.run(
            cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
    log.write(proc.stdout)
    return proc.returncode, proc.stdout


def sr_airway_volume_mL(sr_dir: Path):
    """Extract the 'Airway Volume: N mL' number from a MAP SR dir (raw
    bytes — the SR stores the narrative as literal text)."""
    dcms = sorted(sr_dir.glob("*.dcm"))
    if len(dcms) != 1:
        raise RuntimeError(f"expected exactly 1 SR .dcm under {sr_dir}, "
                           f"found {len(dcms)}")
    m = SR_VOLUME_RE.search(dcms[0].read_bytes())
    if not m:
        raise RuntimeError(f"no 'Airway Volume: N mL' text in {dcms[0]}")
    return float(m.group(1))


def run_fast_app(model_list_env, out_dir: Path):
    """Run the fast app E2E into out_dir. Returns (rc, log text, parsed
    run/ensemble lists)."""
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("HOLOSCAN_MODEL_LIST", None)
    if model_list_env is not None:
        env["HOLOSCAN_MODEL_LIST"] = model_list_env
    env["PYTHONUNBUFFERED"] = "1"
    log_path = out_dir.parent / f"{out_dir.name}.log"
    rc, text = run_subprocess(
        [PY, "my_app", "-i", str(INPUT_DIR), "-m", str(MODEL_DIR),
         "-o", str(out_dir)],
        cwd=FAST_APP_ROOT, env=env, log_path=log_path)
    m = LIST_RE.search(text)
    run_list = ensemble_list = None
    if m:
        run_list = json.loads(m.group(1))
        ensemble_list = json.loads(m.group(2))
    return rc, text, run_list, ensemble_list, str(log_path)


def pixel_diff(fast_dir: Path, oracle: str, json_path: Path):
    """Run pixel_diff.py fast-SEG vs oracle; return (rc, report)."""
    rc, _ = run_subprocess(
        [PY, str(PIXEL_DIFF), str(fast_dir), oracle, "--json", str(json_path)],
        cwd=FAST_APP_ROOT, env=os.environ.copy(),
        log_path=json_path.parent / "pixel_diff.out")
    with open(json_path) as f:
        return rc, json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scratch", default=str(SCRATCH))
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args(argv)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    # Holoscan recommends a 32 MB stack; raise it here so forked app
    # subprocesses inherit it (same pattern as baseline_benchmark.py).
    try:
        resource.setrlimit(resource.RLIMIT_STACK,
                           (32 * 1024 * 1024, resource.RLIM_INFINITY))
    except (ValueError, OSError) as e:
        print(f"[warn] could not raise stack limit: {e}")

    results = []
    for row in GATE_ROWS:
        out_dir = scratch / row["row"]
        print("=" * 72)
        print(f"gate: {row['row']}  (HOLOSCAN_MODEL_LIST="
              f"{row['model_list_env'] or '<unset>'}) vs {row['oracle']}")
        print("=" * 72)
        r = {"row": row["row"],
             "model_list": row["model_list_env"],
             "oracle": row["oracle"]}

        # 1. fast app E2E
        rc, _, run_list, ensemble_list, log_path = run_fast_app(
            row["model_list_env"], out_dir)
        r["fast_exit"] = rc
        r["fast_log"] = log_path
        r["logged_run_model_list"] = run_list
        r["logged_ensemble_model_list"] = ensemble_list
        r["lists_match"] = (run_list == row["expected_run"]
                            and ensemble_list == row["expected_ensemble"])
        if rc != 0:
            r.update({"pass": False, "fail_reasons": ["fast app exit "
                                                      f"{rc} (see {log_path})"]})
            results.append(r)
            print(f"  fast app exit {rc} — FAIL (log: {log_path})")
            continue
        if not r["lists_match"]:
            print(f"  LIST MISMATCH: run={run_list} ensemble={ensemble_list} "
                  f"(expected {row['expected_run']} / "
                  f"{row['expected_ensemble']})")

        # 2. pixel diff
        json_path = scratch / f"{row['row']}.pixel_diff.json"
        pd_rc, pd = pixel_diff(out_dir, row["oracle"], json_path)
        r["pixel_diff"] = {
            "byte_identity_pct": pd["pixels"]["byte_identity_pct"],
            "differing_voxels": pd["pixels"]["differing_voxels"],
            "iou": pd["pixels"]["iou"],
            "fast_voxels": pd["a"]["voxels"],
            "oracle_voxels": pd["b"]["voxels"],
            "geometry_match": pd["geometry"]["match"],
            "pass": pd["pass"],
            "fail_reasons": pd["fail_reasons"],
        }
        # provenance: checksum of the oracle SEG (the oracle bytes are
        # gitignored per repo convention — the JSON is their record)
        oracle_segs = sorted((Path(row["oracle"]) / "SEG").glob("*.dcm"))
        r["oracle_seg_sha256"] = (hashlib.sha256(
            oracle_segs[0].read_bytes()).hexdigest()
            if oracle_segs else None)
        r["oracle_seg_file"] = str(oracle_segs[0]) if oracle_segs else None
        print(f"  pixel_diff: {pd['pixels']['byte_identity_pct']:.5f}% "
              f"byte-identity, {pd['pixels']['differing_voxels']} differing "
              f"voxels, fast={pd['a']['voxels']} oracle={pd['b']['voxels']} "
              f"voxels -> {'PASS' if pd['pass'] else 'FAIL'}")

        # 3. SR airway volume
        try:
            sr_fast = sr_airway_volume_mL(out_dir / "SR")
            sr_oracle = sr_airway_volume_mL(Path(row["oracle"]) / "SR")
            sr_delta = (100.0 * abs(sr_fast - sr_oracle) / sr_oracle
                        if sr_oracle else (0.0 if sr_fast == sr_oracle
                                            else float("inf")))
            sr_ok = sr_delta <= 0.1
        except Exception as e:  # noqa: BLE001
            sr_fast = sr_oracle = sr_delta = None
            sr_ok = False
            print(f"  SR compare error: {e}")
        r["sr_fast"] = sr_fast
        r["sr_oracle"] = sr_oracle
        r["sr_delta_pct"] = sr_delta
        r["sr_ok"] = sr_ok
        print(f"  SR airway volume: fast={sr_fast} mL oracle={sr_oracle} mL "
              f"delta={sr_delta}% -> {'PASS' if sr_ok else 'FAIL'}")

        fail = []
        if not r["lists_match"]:
            fail.append("logged run/ensemble model lists do not match the "
                        "expected pair for this gate row")
        if not pd["pass"]:
            fail.extend(pd["fail_reasons"])
        if not sr_ok:
            fail.append(f"SR airway volume delta {sr_delta}% > 0.1%")
        r["pass"] = not fail
        r["fail_reasons"] = fail
        results.append(r)
        print(f"  gate {row['row']}: {'PASS' if r['pass'] else 'FAIL'}")

    # sanity: the bundle must actually ENSEMBLE (its SEG differs from the
    # fullres-only SEG — different configurations; D-06 class ~2447 vs
    # ~3655 post-CC voxels)
    sanity = {}
    by_row = {r["row"]: r for r in results}
    if (by_row.get("bundle", {}).get("fast_exit") == 0
            and by_row.get("fullres_only", {}).get("fast_exit") == 0):
        rc, pd = pixel_diff(scratch / "bundle", str(scratch / "fullres_only"),
                            scratch / "bundle_vs_fullres.sanity.json")
        # rc is 1 when they differ (diff beyond the pass tolerance) — that
        # is the EXPECTED outcome here; we only need the voxel evidence.
        sanity = {
            "bundle_vs_fullres_differing_voxels":
                pd["pixels"]["differing_voxels"],
            "bundle_voxels": pd["a"]["voxels"],
            "fullres_only_voxels": pd["b"]["voxels"],
            "ok": pd["pixels"]["differing_voxels"] > 0,
        }
        print(f"  sanity: bundle vs fullres-only — "
              f"{pd['pixels']['differing_voxels']} differing voxels "
              f"(bundle={pd['a']['voxels']}, fullres={pd['b']['voxels']}) "
              f"-> {'OK (bundle actually ensembles)' if sanity['ok'] else 'WARN'}")

    # 5. residency (bundle configuration: runtime runs with HOLOSCAN_MODEL_LIST
    #    unset = the multi-fragment default)
    res_env = os.environ.copy()
    res_env.pop("HOLOSCAN_MODEL_LIST", None)
    res_env["PYTHONUNBUFFERED"] = "1"
    static_rc, _ = run_subprocess(
        [PY, str(GPU_RESIDENCY), "--static"], cwd=FAST_APP_ROOT,
        env=res_env, log_path=scratch / "residency_static.out")
    runtime_rc, _ = run_subprocess(
        [PY, str(GPU_RESIDENCY), "--runtime",
         "--output", str(scratch / "residency_out")],
        cwd=FAST_APP_ROOT, env=res_env, log_path=scratch / "residency_runtime.out")
    residency = {"static": "PASS" if static_rc == 0 else "FAIL",
                 "runtime": "PASS" if runtime_rc == 0 else "FAIL",
                 "static_log": str(scratch / "residency_static.out"),
                 "runtime_log": str(scratch / "residency_runtime.out"),
                 "note": "runtime runs the multi-fragment BUNDLE configuration "
                         "(HOLOSCAN_MODEL_LIST unset = reference default); "
                         "preprocess_operator.py is the deliberate D-13 "
                         "allow-list entry (Plan 01); postprocess_operator.py "
                         "is the only exactly-once final boundary"}
    print(f"  residency: static={residency['static']} "
          f"runtime={residency['runtime']}")

    deviations = [
        {"id": "TEST-005-2d",
         "text": "2d config blocked-on-model (D-01/D-03): the bundle has no "
                 "2d model; TEST-005 counts as met-with-deviation, same "
                 "pattern as the Phase 0 corpus deviation. Fragment wiring is "
                 "config-generic (D-02) — a real 2d model is a test, not a "
                 "code change (D-04)."},
        {"id": "TEST-01-corpus",
         "text": "Single airway dev-study corpus (TEST-01 deviation, carried "
                 "from Phase 0/1); the >=5-CT-study re-run remains a "
                 "deferred final gate."},
    ]

    all_pass = (all(r["pass"] for r in results)
                and static_rc == 0 and runtime_rc == 0)
    report = {
        "phase": "2-gpu-acceleration",
        "plan": "05",
        "gates": results,
        "residency": residency,
        "sanity": sanity,
        "deviations": deviations,
        "fixes": [],
        "all_gates_pass": all_pass,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("PHASE 2 GATE SUMMARY")
    print("=" * 72)
    print(f"{'gate':<15} {'identity%':>10} {'diff vox':>9} {'sr delta%':>10} "
          f"{'residency n/a':>13}  result")
    for r in results:
        pd = r.get("pixel_diff") or {}
        sr_d = r.get("sr_delta_pct")
        sr_col = f"{sr_d:.4f}" if sr_d is not None else "-"
        id_col = f"{pd.get('byte_identity_pct', 0.0):.5f}"
        dv_col = f"{pd.get('differing_voxels', '-')}"
        print(f"{r['row']:<15} {id_col:>10} {dv_col:>9} {sr_col:>10} "
              f"{'':>13}  {'PASS' if r['pass'] else 'FAIL'}")
    print(f"residency: static={residency['static']} runtime={residency['runtime']}")
    if sanity:
        print(f"sanity (bundle ensembles): {sanity}")
    print(f"report: {args.report}")
    print(f"ALL GATES: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
