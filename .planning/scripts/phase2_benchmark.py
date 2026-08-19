#!/usr/bin/env python3
"""
Phase 2 benchmark for the FAST app (cchmc-nnunet-fast) — two-bar, per-operator.

Extends the Phase 0/1 pattern in baseline_benchmark.py (fresh subprocess per
rep so model-load cold start is included — the clinical single-study usage;
sitecustomize logging shim; raised 32 MB stack limit).

Scopes (the resolved D-18 two-bar report — 02-CONTEXT.md):
  fullres — HOLOSCAN_MODEL_LIST=3d_fullres: the SAME-SCOPE bar vs Phase 1's
            61.8 s single-config fullres E2E (the positive-improvement bar).
  bundle  — HOLOSCAN_MODEL_LIST UNSET (reference default:
            3d_fullres + 3d_lowres + 3d_cascade_fullres, ensemble over
            fullres + cascade_fullres): the HEADLINE bar vs the 169.7 s
            reference baseline (169,747 ± 7,274 ms, .planning/baseline_results.csv).

Structured-log parsing (INFR-005 / RESEARCH Pitfall 9 — Plan 04 made the
fast app's logs config-tagged and app-level, one study_timing_summary per
run regardless of sub-Fragment count):
  timing: {json}                 — one per operator per compute:
                                   {"operator","label","study","start","end",
                                    "start_ns","end_ns","duration_ms"[,"config"]}
  study_timing_summary: {json}   — {"study","operators","total_ms","n_records"}
Operator naming: preprocess_<cfg> / inference_<cfg> / postresample_<cfg> /
ensemble_average / postprocess / write_seg / write_sr / write_sc / model_load.

CSV columns (one row per rep; the header is the UNION over both scopes so the
script is append-safe across the two scope invocations):
  scope,study,rep,warmup,total_ms,
  preprocess_ms_<cfg>,inference_ms_<cfg>,postresample_ms_<cfg>  (per config),
  ensemble_ms,postprocess_ms,write_ms,
  speedup_vs_61.8s,speedup_vs_169.7s,ok
Speedup constants:
  61,800 ms   — Phase 1 fast-app center (baseline-2026-08-18.csv, 61.2–62.2 s)
  169,747 ms  — reference baseline mean (baseline_results.csv, n=4)
A `summary` row (rep="mean±std", warmup="summary") per scope is appended after
each scope run (warmups excluded from the statistics).

Usage:
    /tmp/monai-env/.venv/bin/python .planning/scripts/phase2_benchmark.py --scope fullres
    /tmp/monai-env/.venv/bin/python .planning/scripts/phase2_benchmark.py --scope bundle
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

# D-18 baseline constants (documented above)
P1_CENTER_MS = 61800.0        # Phase 1 fast-app fullres-only E2E center
REFERENCE_MS = 169747.0       # reference app baseline mean

# The three 3D configs (2d blocked-on-model, D-01/D-03)
CONFIGS = ["3d_fullres", "3d_lowres", "3d_cascade_fullres"]

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
    fields = ["scope", "study", "rep", "warmup", "total_ms"]
    for cfg in CONFIGS:
        for stage in STAGES:
            fields.append(f"{stage}_ms_{cfg}")
    fields += ["ensemble_ms", "postprocess_ms", "write_ms",
               "speedup_vs_61.8s", "speedup_vs_169.7s", "ok"]
    return fields


def parse_timing_logs(stdout_text):
    """Parse the fast app's structured logs into per-operator durations (ms).

    Returns (operators: {operator_name: duration_ms}, n_timing_records,
    n_records_from_summary).
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


def build_row(scope, study, rep, is_warmup, total_ms, operators, ok):
    row = {
        "scope": scope,
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


def run_once(study_dir, model, output_root, python, scope):
    """One fresh-subprocess app run. Returns (total_ms, operators, ok, tail, n_timing, n_records_summary)."""
    out_dir = tempfile.mkdtemp(prefix=f"phase2_{scope}_", dir=output_root)
    env = os.environ.copy()
    shim_dir = tempfile.mkdtemp(prefix="logshim_")
    with open(os.path.join(shim_dir, "sitecustomize.py"), "w") as f:
        f.write(SITECUSTOMIZE)
    env["PYTHONPATH"] = shim_dir + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
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
    proc = subprocess.run(
        cmd, cwd=str(FAST_APP_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    total_ms = (time.monotonic() - t0) * 1000.0
    shutil.rmtree(shim_dir, ignore_errors=True)

    operators, n_timing, n_records_summary = parse_timing_logs(proc.stdout)
    ok = proc.returncode == 0
    tail = "\n".join(proc.stdout.splitlines()[-15:])
    return total_ms, operators, ok, tail, n_timing, n_records_summary


def summarize_numeric(rows, header):
    """mean ± std over the numeric columns of the measured rows."""
    out = {}
    for col in header:
        if col in ("scope", "study", "rep", "warmup", "ok"):
            continue
        vals = [float(r[col]) for r in rows if r[col] not in ("", None)]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[col] = (mean, std)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", required=True, choices=["fullres", "bundle"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--output-csv", default=str(REPO_ROOT / ".planning" / "benchmarks" / "phase2_results.csv"))
    ap.add_argument("--study-dir", default=str(REPO_ROOT / "testdata" / "airway_input"))
    ap.add_argument("--model", default=str(REPO_ROOT / "examples" / "apps" / "cchmc_nnunet_fifteen_ckpt_app" / "models"))
    ap.add_argument("--python", default="/tmp/monai-env/.venv/bin/python")
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args()

    study_name = os.path.basename(os.path.normpath(args.study_dir))
    output_root = args.output_root or tempfile.mkdtemp(prefix="phase2_root_")
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # Append-safe CSV: reuse the existing header if the file exists.
    existing_rows = []
    if os.path.exists(args.output_csv):
        with open(args.output_csv, newline="") as f:
            existing_rows = list(csv.DictReader(f))
    header = csv_header()

    rows = []
    for rep in range(args.warmups + args.reps):
        is_warmup = rep < args.warmups
        print(f"[{args.scope} rep {rep + 1}/{args.warmups + args.reps}] {'WARMUP' if is_warmup else 'MEASURED'} run start", flush=True)
        total_ms, operators, ok, tail, n_timing, n_records_summary = run_once(
            args.study_dir, args.model, output_root, args.python, args.scope
        )
        print(f"[{args.scope} rep {rep + 1}] total={total_ms:,.0f} ms ok={ok} timing_records={n_timing} "
              f"summary_n_records={n_records_summary}", flush=True)
        if not ok:
            print("--- log tail ---")
            print(tail)
        # consistency: operator sum vs in-study portion of total
        op_sum = sum(operators.values()) - operators.get("model_load", 0.0)
        if ok:
            print(f"[{args.scope} rep {rep + 1}] in-study operator sum={op_sum:,.0f} ms "
                  f"({op_sum / total_ms * 100:.0f}% of wall total)")
        row = build_row(args.scope, study_name, rep + 1, is_warmup, total_ms, operators, ok)
        rows.append(row)
        if not ok:
            break

    measured = [r for r in rows if r["warmup"] == "false"]
    stats = summarize_numeric(measured, header)

    # Console summary (mean ± std, warmups excluded)
    print(f"\n=== {args.scope} scope: measured mean ± std (n={len(measured)}) ===")
    for col in header:
        if col in ("scope", "study", "rep", "warmup", "ok"):
            continue
        if col in stats:
            mean, std = stats[col]
            print(f"  {col}: {mean:,.1f} ± {std:,.1f}")
    if "total_ms" in stats:
        mean, std = stats["total_ms"]
        print(f"  speedup vs Phase 1 61.8 s : {P1_CENTER_MS / mean:.3f}")
        print(f"  speedup vs reference 169.7 s: {REFERENCE_MS / mean:.3f}")

    # summary row (rep="mean±std")
    summary_row = {f: "" for f in header}
    summary_row.update({"scope": args.scope, "study": study_name,
                        "rep": "mean±std", "warmup": "summary", "ok": "true"})
    for col, (mean, std) in stats.items():
        summary_row[col] = f"{mean:,.1f}±{std:,.1f}"

    # Append-safe write: rewrite file with prior rows + this scope's rows.
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(existing_rows)
        w.writerows(rows)
        w.writerow(summary_row)
    print(f"\nCSV written: {args.output_csv} "
          f"(+{len(rows)} rep rows, +1 summary row; {len(existing_rows)} prior rows kept)")


if __name__ == "__main__":
    main()
