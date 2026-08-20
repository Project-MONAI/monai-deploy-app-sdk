#!/usr/bin/env python3
"""
Phase 3 close-out benchmark for the FAST app (cchmc-nnunet-fast) — the 2x2
matrix (D-18 two-bar convention, extended per 03-RESEARCH Measurement
Architecture + 3-optimization-05-PLAN.md).

Copy base: phase2_benchmark.py (NOT rebuilt — D-25). Methodology is the
Phase 2 one, unchanged:
  - fresh subprocess per rep (model-load cold start included — the clinical
    single-study usage), sitecustomize logging shim, 32 MB stack rlimit,
    1 warmup + 3 measured reps per cell (warmups excluded from statistics),
    per-rep `timeout` (default 900 s) so a wedged run cannot hang the matrix,
    parses the config-tagged `timing: {json}` logs BY OPERATOR NAME + the
    single app-level study_timing_summary (Pitfall 6-safe: no record-order
    assumptions — timing-record order is nondeterministic under the
    concurrent scheduler).
  - GPU pinned via CUDA_VISIBLE_DEVICES (Pitfall 7) and recorded in the
    `gpu` CSV column (provenance).

The 2x2 matrix (one cell per scope; both flags set EXPLICITLY per cell —
never relying on defaults):
  cell                    HOLOSCAN_CONCURRENT_FRAGMENTS  HOLOSCAN_GPU_RESAMPLE
  conc-resample-off       1                              0
  conc-resample-on        1                              1
  serial-resample-off     0                              0
  serial-resample-on      0                              1
Scopes (D-18, same as Phase 2):
  fullres — HOLOSCAN_MODEL_LIST=3d_fullres (same-scope bar vs Phase 1 61.8 s
            and Phase 2 57.14 s).
  bundle  — HOLOSCAN_MODEL_LIST UNSET (reference default: fullres + lowres +
            cascade_fullres, ensemble over fullres + cascade_fullres;
            headline bar vs the 169,747 ms reference baseline).

Plan-04 verdict note: the flag shipped ON (amended D-22b — gate green,
per-tensor 100.0000% accuracy), so ALL four cells are runnable; the ON
column is HOLOSCAN_GPU_RESAMPLE=1 and the OFF column = HOLOSCAN_GPU_RESAMPLE=0
(the scipy/skimage CPU reference path, no longer the default).

CSV columns: phase2's columns (identical names — the Phase 3 'before' is
.planning/benchmarks/phase2_results.csv) plus `cell` and `gpu`:
  scope,cell,gpu,study,rep,warmup,total_ms,
  preprocess_ms_<cfg>,inference_ms_<cfg>,postresample_ms_<cfg>  (per config),
  ensemble_ms,postprocess_ms,write_ms,
  speedup_vs_61.8s,speedup_vs_169.7s,ok
Speedup constants (D-18):
  61,800 ms   — Phase 1 fast-app center (baseline-2026-08-18.csv, 61.2-62.2 s)
  169,747 ms  — reference baseline mean (baseline_results.csv, n=4)
Per-cell `summary` rows (rep="mean±std", warmup="summary", warmups excluded)
are appended after each cell. The file is append-safe: rows for cells
already present are kept; a re-run of a cell appends fresh rows after.

Usage (one invocation runs the WHOLE matrix):
    /tmp/monai-env/.venv/bin/python .planning/scripts/phase3_benchmark.py \
        --out .planning/benchmarks/phase3_results.csv --gpu 0
Optional subset:
    ... --scopes bundle --cells conc-resample-on serial-resample-on
"""

import argparse
import csv
import json
import os
import re
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FAST_APP_ROOT = REPO_ROOT / "examples" / "apps" / "cchmc-nnunet-fast"

# D-18 baseline constants (documented above; identical to phase2_benchmark.py)
P1_CENTER_MS = 61800.0        # Phase 1 fast-app fullres-only E2E center
REFERENCE_MS = 169747.0       # reference app baseline mean

# The three 3D configs (2d blocked-on-model, D-01/D-03)
CONFIGS = ["3d_fullres", "3d_lowres", "3d_cascade_fullres"]

# The 2x2 matrix: cell -> (HOLOSCAN_CONCURRENT_FRAGMENTS, HOLOSCAN_GPU_RESAMPLE)
# Explicit values per cell — never rely on defaults (both flags now default ON).
CELLS = {
    "conc-resample-off": ("1", "0"),
    "conc-resample-on": ("1", "1"),
    "serial-resample-off": ("0", "0"),
    "serial-resample-on": ("0", "1"),
}

TIMING_RE = re.compile(r"timing: (\{.*\})\s*$")
SUMMARY_RE = re.compile(r"study_timing_summary: (\{.*\})\s*$")

SITECUSTOMIZE = """
import logging
# Only configure if nothing else has (app may configure later; basicConfig is a no-op after)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
"""

STAGES = ("preprocess", "inference", "postresample")


def csv_header():
    fields = ["scope", "cell", "gpu", "study", "rep", "warmup", "total_ms"]
    for cfg in CONFIGS:
        for stage in STAGES:
            fields.append(f"{stage}_ms_{cfg}")
    fields += ["ensemble_ms", "postprocess_ms", "write_ms",
               "speedup_vs_61.8s", "speedup_vs_169.7s", "ok"]
    return fields


def parse_timing_logs(stdout_text):
    """Parse the fast app's structured logs into per-operator durations (ms).

    Parses BY OPERATOR NAME — no record-order assumptions (Pitfall 6: under
    the concurrent scheduler the timing-record order follows thread
    scheduling, not DAG order). Returns (operators: {name: duration_ms},
    n_timing_records, n_records_from_summary).
    """
    operators = {}
    n_timing = 0
    n_records_summary = None
    for line in stdout_text.splitlines():
        m = TIMING_RE.search(line)
        if m:
            try:
                rec = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            operators[rec["operator"]] = rec["duration_ms"]
            n_timing += 1
            continue
        m = SUMMARY_RE.search(line)
        if m:
            try:
                summary = json.loads(m.group(1))
                n_records_summary = summary.get("n_records")
            except json.JSONDecodeError:
                pass
    return operators, n_timing, n_records_summary


def build_row(scope, cell, gpu, study, rep, is_warmup, total_ms, operators, ok):
    row = {
        "scope": scope,
        "cell": cell,
        "gpu": str(gpu),
        "study": study,
        "rep": str(rep),
        "warmup": str(is_warmup).lower(),
        "total_ms": f"{total_ms:.1f}",
    }
    for cfg in CONFIGS:
        for stage in STAGES:
            row[f"{stage}_ms_{cfg}"] = (
                f"{operators[f'{stage}_{cfg}']:.1f}" if f"{stage}_{cfg}" in operators else ""
            )
    row["ensemble_ms"] = f"{operators['ensemble_average']:.1f}" if "ensemble_average" in operators else ""
    row["postprocess_ms"] = f"{operators['postprocess']:.1f}" if "postprocess" in operators else ""
    writers = [operators[k] for k in ("write_seg", "write_sr", "write_sc") if k in operators]
    row["write_ms"] = f"{sum(writers):.1f}" if writers else ""
    row["speedup_vs_61.8s"] = f"{P1_CENTER_MS / total_ms:.3f}" if total_ms > 0 else ""
    row["speedup_vs_169.7s"] = f"{REFERENCE_MS / total_ms:.3f}" if total_ms > 0 else ""
    row["ok"] = str(ok).lower()
    return row


def run_once(study_dir, model, output_root, python, scope, cell, gpu, timeout_s):
    """One fresh-subprocess app run. Returns (total_ms, operators, ok, tail, n_timing, n_records_summary)."""
    out_dir = tempfile.mkdtemp(prefix=f"phase3_{scope}_{cell}_", dir=output_root)
    env = os.environ.copy()
    shim_dir = tempfile.mkdtemp(prefix="logshim_")
    with open(os.path.join(shim_dir, "sitecustomize.py"), "w") as f:
        f.write(SITECUSTOMIZE)
    env["PYTHONPATH"] = shim_dir + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    # GPU pinning (Pitfall 7) — recorded in the CSV `gpu` column.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # The two matrix flags — EXPLICIT per cell, never defaults.
    env["HOLOSCAN_CONCURRENT_FRAGMENTS"] = CELLS[cell][0]
    env["HOLOSCAN_GPU_RESAMPLE"] = CELLS[cell][1]
    # HOLOSCAN_MODEL_LIST: pinned single config for the same-scope bar;
    # UNSET for the bundle (reference default = the 3-config run list).
    env.pop("HOLOSCAN_MODEL_LIST", None)
    if scope == "fullres":
        env["HOLOSCAN_MODEL_LIST"] = "3d_fullres"

    cmd = [python, "-m", "my_app", "--input", study_dir, "--model", model,
           "--output", out_dir]

    # Holoscan recommends a 32 MB stack; raise in this process so the
    # forked subprocess inherits it (baseline_benchmark.py pattern).
    try:
        resource.setrlimit(resource.RLIMIT_STACK, (32 * 1024 * 1024, resource.RLIM_INFINITY))
    except (ValueError, OSError) as e:
        print(f"[warn] could not raise stack limit: {e}")

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=str(FAST_APP_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=timeout_s,
        )
        total_ms = (time.monotonic() - t0) * 1000.0
    except subprocess.TimeoutExpired as e:
        total_ms = (time.monotonic() - t0) * 1000.0
        stdout_text = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        proc = type("P", (), {"returncode": 124, "stdout": stdout_text})()
        print(f"[{scope}/{cell}] rep TIMED OUT after {timeout_s}s", flush=True)
    finally:
        shutil.rmtree(shim_dir, ignore_errors=True)

    operators, n_timing, n_records_summary = parse_timing_logs(proc.stdout)
    ok = proc.returncode == 0
    tail = "\n".join(proc.stdout.splitlines()[-15:])
    return total_ms, operators, ok, tail, n_timing, n_records_summary


def summarize_numeric(rows, header):
    """mean ± std over the numeric columns of the measured rows."""
    out = {}
    for col in header:
        if col in ("scope", "cell", "gpu", "study", "rep", "warmup", "ok"):
            continue
        vals = [float(r[col]) for r in rows if r.get(col) not in ("", None)]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[col] = (mean, std)
    return out


def run_cell(args, scope, cell, study_name, output_root, header):
    """Run one (scope, cell): warmups + measured reps, append rows + summary row."""
    env_flag_summary = f"concurrent={CELLS[cell][0]} resample={CELLS[cell][1]}"
    existing_rows = []
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            existing_rows = list(csv.DictReader(f))

    rows = []
    for rep in range(args.warmups + args.reps):
        is_warmup = rep < args.warmups
        tag = "WARMUP" if is_warmup else "MEASURED"
        print(f"[{scope}/{cell} rep {rep + 1}/{args.warmups + args.reps}] {tag} "
              f"({env_flag_summary}, gpu={args.gpu}) run start", flush=True)
        total_ms, operators, ok, tail, n_timing, n_records_summary = run_once(
            args.study_dir, args.model, output_root, args.python,
            scope, cell, args.gpu, args.timeout_s,
        )
        print(f"[{scope}/{cell} rep {rep + 1}] total={total_ms:,.0f} ms ok={ok} "
              f"timing_records={n_timing} summary_n_records={n_records_summary}", flush=True)
        if not ok:
            print("--- log tail ---")
            print(tail)
        op_sum = sum(operators.values()) - operators.get("model_load", 0.0)
        if ok:
            print(f"[{scope}/{cell} rep {rep + 1}] in-study operator sum={op_sum:,.0f} ms "
                  f"({op_sum / total_ms * 100:.0f}% of wall total)")
        row = build_row(scope, cell, args.gpu, study_name, rep + 1, is_warmup,
                        total_ms, operators, ok)
        rows.append(row)
        if not ok:
            print(f"[{scope}/{cell}] run FAILED — stopping this cell", flush=True)
            break

    measured = [r for r in rows if r["warmup"] == "false" and r["ok"] == "true"]
    stats = summarize_numeric(measured, header)

    print(f"\n=== {scope}/{cell}: measured mean ± std (n={len(measured)}) ===")
    for col in header:
        if col in ("scope", "cell", "gpu", "study", "rep", "warmup", "ok"):
            continue
        if col in stats:
            mean, std = stats[col]
            print(f"  {col}: {mean:,.1f} ± {std:,.1f}")
    if "total_ms" in stats:
        mean, std = stats["total_ms"]
        print(f"  speedup vs Phase 1 61.8 s : {P1_CENTER_MS / mean:.3f}")
        print(f"  speedup vs reference 169.7 s: {REFERENCE_MS / mean:.3f}")

    summary_row = {f: "" for f in header}
    summary_row.update({"scope": scope, "cell": cell, "gpu": str(args.gpu),
                        "study": study_name, "rep": "mean±std",
                        "warmup": "summary", "ok": "true" if len(measured) else "false"})
    for col, (mean, std) in stats.items():
        summary_row[col] = f"{mean:,.1f}±{std:,.1f}"

    # Append-safe write: prior rows kept; this cell's rows + summary appended.
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(existing_rows)
        w.writerows(rows)
        w.writerow(summary_row)
    print(f"[{scope}/{cell}] CSV updated: {args.out} "
          f"(+{len(rows)} rep rows, +1 summary row)", flush=True)
    return all(r["ok"] == "true" for r in rows) and len(measured) >= args.reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scopes", nargs="+", default=["fullres", "bundle"],
                    choices=["fullres", "bundle"])
    ap.add_argument("--cells", nargs="+", default=list(CELLS.keys()),
                    choices=list(CELLS.keys()))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--gpu", default="0",
                    help="CUDA_VISIBLE_DEVICES value to pin (recorded in the CSV gpu column)")
    ap.add_argument("--timeout-s", type=int, default=900,
                    help="per-rep wall timeout in seconds (default 900)")
    ap.add_argument("--out", default=str(REPO_ROOT / ".planning" / "benchmarks" / "phase3_results.csv"))
    ap.add_argument("--study-dir", default=str(REPO_ROOT / "testdata" / "airway_input"))
    ap.add_argument("--model", default=str(REPO_ROOT / "examples" / "apps" / "cchmc_nnunet_fifteen_ckpt_app" / "models"))
    ap.add_argument("--python", default="/tmp/monai-env/.venv/bin/python")
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args()

    study_name = os.path.basename(os.path.normpath(args.study_dir))
    output_root = args.output_root or tempfile.mkdtemp(prefix="phase3_root_")
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    header = csv_header()

    # Seed the CSV header if the file does not exist.
    if not os.path.exists(args.out):
        with open(args.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writeheader()

    ok_all = True
    for scope in args.scopes:
        for cell in args.cells:
            cell_ok = run_cell(args, scope, cell, study_name, output_root, header)
            ok_all = ok_all and cell_ok

    print(f"\n=== Phase 3 2x2 matrix {'COMPLETE' if ok_all else 'INCOMPLETE (see above)'} "
          f"(scopes={args.scopes}, cells={args.cells}, gpu={args.gpu}) ===")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
