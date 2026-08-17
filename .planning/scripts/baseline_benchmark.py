#!/usr/bin/env python3
"""
Baseline benchmark for the reference app (cchmc_nnunet_fifteen_ckpt_app).

Task 0.4: runs the current app on the reference corpus, records end-to-end
latency and per-stage timing, and writes a CSV.

Method:
- The app is launched as a subprocess (fresh process per rep, so model-load
  cold start is included — matching real single-study clinical usage).
- A sitecustomize.py shim (via PYTHONPATH) configures Python logging at INFO
  with timestamps so per-stage markers can be parsed from stdout:
    "Loading nnU-Net ensemble models"            → setup begin (model load)
    "Running nnU-Net ensemble inference..."      → inference begin
    "Inference complete, applying postprocessing"→ inference end
    "Calculated Organ Volumes"                   → postprocess end
    "End my_app.app" / process exit              → pipeline end
- Stage boundaries are derived from those markers; total is subprocess wall time.
- N warmup runs precede N measured runs (per ROADMAP risk mitigation:
  3 repetitions, mean ± std reported by the caller).

Usage:
    python baseline_benchmark.py \
        --app-dir <path to cchmc_nnunet_fifteen_ckpt_app> \
        --study-dir <dicom input dir> \
        --model <model root> \
        --output-csv <csv path> \
        [--reps 3] [--warmups 1] [--output-root <scratch dir>]

CSV columns: study,rep,warmup,total_ms,setup_ms,inference_ms,postprocess_ms,write_ms
"""

import argparse
import csv
import logging
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time

SITECUSTOMIZE = """
import logging, os
# Only configure if nothing else has (app may configure later; basicConfig is a no-op after)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
"""

# Two timestamp formats appear in app output:
# 1) Python basicConfig (via sitecustomize shim):  2026-08-17 00:07:37.715 INFO name - msg
# 2) Holoscan C++/entity logger:                   [2026-08-17 00:07:40,267] [INFO] (name) - msg
TS_PY = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})\.(\d{3}) ")
TS_CXX = re.compile(r"^\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3})\]")

MARKERS = {
    "setup_begin": re.compile(r"Loading nnU-Net ensemble models$"),
    "inference_begin": re.compile(r"Running nnU-Net ensemble inference\.\.\.$"),
    "inference_end": re.compile(r"Inference complete, applying postprocessing\.\.\.$"),
    "postprocess_end": re.compile(r"Calculated Organ Volumes"),
    "pipeline_end": re.compile(r"End (my_app\.app|__main__|run)"),
}


def parse_timestamp(line):
    m = TS_PY.match(line) or TS_CXX.match(line)
    if not m:
        return None
    import datetime

    dt = datetime.datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S")
    return dt.timestamp() * 1000.0 + int(m.group(3))


def run_once(app_dir, study_dir, model, output_root, python):
    """Run the app once in a fresh subprocess. Returns (total_ms, stage_ms dict, ok, log_tail, out_dir)."""
    out_dir = tempfile.mkdtemp(prefix="baseline_run_", dir=output_root)
    env = os.environ.copy()
    shim_dir = tempfile.mkdtemp(prefix="logshim_")
    with open(os.path.join(shim_dir, "sitecustomize.py"), "w") as f:
        f.write(SITECUSTOMIZE)
    env["PYTHONPATH"] = shim_dir + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    # Holoscan recommends 32 MB stack to avoid segfaults
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = [
        python,
        "-m",
        "my_app",
        "--input",
        study_dir,
        "--model",
        model,
        "--output",
        out_dir,
    ]
    # Holoscan recommends a 32 MB stack to avoid segfaults; raise it in this process
    # so the forked subprocess inherits it.
    try:
        resource.setrlimit(resource.RLIMIT_STACK, (32 * 1024 * 1024, resource.RLIM_INFINITY))
    except (ValueError, OSError) as e:  # some platforms cannot raise it
        print(f"[warn] could not raise stack limit: {e}")
    t0_wall = time.time()  # epoch seconds, comparable with log timestamps
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=app_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    total_ms = (time.monotonic() - t0) * 1000.0
    shutil.rmtree(shim_dir, ignore_errors=True)

    markers = {}
    for line in proc.stdout.splitlines():
        for key, pat in MARKERS.items():
            if key not in markers and pat.search(line):
                ts = parse_timestamp(line)
                if ts is not None:
                    markers[key] = ts

    def span(a, b):
        return (markers[b] - markers[a]) if a in markers and b in markers else None

    t0_ms_local = t0_wall * 1000.0  # process start in the log-timestamp epoch
    stages = {
        # includes python startup + DICOM load + model load
        "setup_ms": markers["inference_begin"] - t0_ms_local if "inference_begin" in markers else None,
        "inference_ms": span("inference_begin", "inference_end"),
        "postprocess_ms": span("inference_end", "postprocess_end"),
        "write_ms": span("postprocess_end", "pipeline_end"),
    }
    ok = proc.returncode == 0
    tail = "\n".join(proc.stdout.splitlines()[-15:])
    return total_ms, stages, ok, tail, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True)
    ap.add_argument("--study-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--output-root", default=None, help="scratch dir for per-run outputs (default: tempdir)")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    study_name = os.path.basename(os.path.normpath(args.study_dir))
    output_root = args.output_root or tempfile.mkdtemp(prefix="baseline_root_")
    os.makedirs(output_root, exist_ok=True)

    rows = []
    for rep in range(args.warmups + args.reps):
        is_warmup = rep < args.warmups
        print(f"[rep {rep + 1}/{args.warmups + args.reps}] {'WARMUP' if is_warmup else 'MEASURED'} run start")
        total_ms, stages, ok, tail, out_dir = run_once(
            args.app_dir, args.study_dir, args.model, output_root, args.python
        )
        print(f"[rep {rep + 1}] total={total_ms:,.0f} ms ok={ok} stages={stages} out={out_dir}")
        if not ok:
            print("--- log tail ---")
            print(tail)
        rows.append(
            {
                "study": study_name,
                "rep": rep + 1,
                "warmup": str(is_warmup).lower(),
                "total_ms": f"{total_ms:.1f}",
                "setup_ms": f"{stages['setup_ms']:.1f}" if stages["setup_ms"] is not None else "",
                "inference_ms": f"{stages['inference_ms']:.1f}" if stages["inference_ms"] is not None else "",
                "postprocess_ms": f"{stages['postprocess_ms']:.1f}" if stages["postprocess_ms"] is not None else "",
                "write_ms": f"{stages['write_ms']:.1f}" if stages["write_ms"] is not None else "",
                "ok": str(ok).lower(),
            }
        )
        if not ok:
            break

    fields = ["study", "rep", "warmup", "total_ms", "setup_ms", "inference_ms", "postprocess_ms", "write_ms", "ok"]
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV written: {args.output_csv}")

    # mean ± std over measured (non-warmup) runs
    measured = [float(r["total_ms"]) for r in rows if r["warmup"] == "false"]
    if measured:
        mean = sum(measured) / len(measured)
        var = sum((x - mean) ** 2 for x in measured) / max(1, len(measured) - 1)
        print(f"MEASURED total_ms: mean={mean:,.1f} std={var ** 0.5:,.1f} n={len(measured)}")


if __name__ == "__main__":
    main()
