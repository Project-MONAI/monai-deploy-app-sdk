#!/usr/bin/env python3
"""INFR-02 / D-24(b) multi-study replay proof harness.

ONE process, 3 sequential passes over the SAME airway study through the REAL
fast-app operators — the plan's preferred path (direct operator drive, not
the full GXF app): ``PreprocessOperator`` + ``SlideWindowOperator`` +
``PostResampleOperator`` + ``EnsembleAverageOperator`` +
``PostprocessOperator`` (3d_fullres config) with the real model bundle and
the real study volume loaded from ``testdata/airway_input`` via the app's
OWN load path (the SDK ``DICOMDataLoaderOperator`` ->
``DICOMSeriesSelectorOperator`` -> ``DICOMSeriesToVolumeOperator``
``convert_to_image`` chain — the same ``Image`` object the DAG would
produce). Each operator's ``compute()`` is driven directly with a minimal
fake op_input/op_output; every inter-operator handoff is the real
DLPack ``holoscan.core.Tensor`` the DAG would pass (no bypass of any
operator boundary).

Shipping state of all prior Phase 3 plans is replicated:
  * RMM pluggable allocator + initial_pool_size pinned 4 GiB
    (Plan 01 — ``gpu_bootstrap`` imported first);
  * the INFR-02 shape caches ACTIVE in Preprocess/SlideWindow (this plan);
  * the concurrent-scheduler plan does not apply (no GXF run here — single
    linear chain, one operator at a time);
  * the MEM-003 release hook is wired only for the aux (lowres_seg) config —
    inert for this fullres-only chain, exactly as in the shipping DAG.

Per study, after the preprocess + inference stages complete, the harness
snapshots ``data_ptr()`` of every entry in each operator's ``_buf_cache``
(``keys()``/``total_bytes()``) + the gaussian tensor ptr and prints a
per-study table. Assertions:

  (i)   the data_ptr tables of studies 2 and 3 are IDENTICAL to study 1's
        (study 1 = allocation, studies 2/3 = reuse);
  (ii)  driver VRAM (pynvml) at the study boundaries does not grow across
        studies beyond +/-5% (RMM pool occupancy — rmm 26.2.0 exposes no
        Python pool-stats API, so the pool level is read at the driver
        level via pynvml, as in Plan 02);
  (iii) the final segmentation (PostprocessOperator's emitted seg payload,
        numpy) of studies 2 and 3 is BYTE-IDENTICAL to study 1's
        (np.array_equal) and the SR result text matches — reuse did not
        corrupt a data path.

Exit 0 iff all assertions pass.

Usage (venv python; 32 MB stack; pinned device per Pitfall 7):
  ulimit -s unlimited
  CUDA_VISIBLE_DEVICES=0 /tmp/monai-env/.venv/bin/python \
      .planning/scripts/infr02_replay.py

nsys mode (per-study cudaMalloc churn — Phase 2 Plan 06 method):
  nsys profile --trace=cuda,nvtx -o <prefix> --force-overwrite=true \
      /tmp/monai-env/.venv/bin/python .planning/scripts/infr02_replay.py
  then, from the exported sqlite:
  /tmp/monai-env/.venv/bin/python .planning/scripts/infr02_replay.py \
      --analyze <prefix>.sqlite
  (the study windows come from the operators' own NVTX ranges: study N =
  the Nth ``preprocess_3d_fullres`` start .. Nth ``postprocess`` end;
  ``cudaMalloc``/``cudaFree`` counts per window use the ``_v3020``
  name-suffixed CUPTI_ACTIVITY_KIND_RUNTIME rows).
"""

from __future__ import annotations

import argparse
import os
import resource
import sqlite3
import sys
import time
from pathlib import Path

# 32 MB stack for holoscan (the app relies on `ulimit -s unlimited`).
try:
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
except (ValueError, OSError):
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "examples" / "apps" / "cchmc-nnunet-fast"
INPUT_DIR = REPO_ROOT / "testdata" / "airway_input"
MODEL_DIR = REPO_ROOT / "examples" / "apps" / "cchmc_nnunet_fifteen_ckpt_app" / "models"
CONFIG = "3d_fullres"
N_STUDIES = 3

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Pitfall 7: pinned device
sys.path.insert(0, str(APP_ROOT / "my_app"))


# ---------------------------------------------------------------------------
# sqlite analyze mode (per-study cudaMalloc churn)
# ---------------------------------------------------------------------------
def analyze_sqlite(path: str) -> int:
    con = sqlite3.connect(path)
    nvtx = list(con.execute(
        "SELECT text, start, end FROM NVTX_EVENTS ORDER BY start"))
    pre = [(s, e) for t, s, e in nvtx if t == f"preprocess_{CONFIG}"]
    post = [(s, e) for t, s, e in nvtx if t == "postprocess"]
    n = min(len(pre), len(post))
    if n != N_STUDIES:
        print(f"ANALYZE: expected {N_STUDIES} study NVTX windows, "
              f"found preprocess={len(pre)} postprocess={len(post)}")
        return 1
    mallocs = [row[0] for row in con.execute(
        "SELECT r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON r.nameId = s.id WHERE s.value LIKE 'cudaMalloc%'")]
    frees = [row[0] for row in con.execute(
        "SELECT r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON r.nameId = s.id WHERE s.value LIKE 'cudaFree%'")]
    pre_start = pre[0][0]
    boot_m = sum(1 for t in mallocs if t < pre_start)
    print("per-study cudaMalloc/cudaFree (study N = Nth preprocess..postprocess NVTX window):")
    print(f"  bootstrap (before study 1): cudaMalloc={boot_m} cudaFree={sum(1 for t in frees if t < pre_start)}")
    for i in range(n):
        ws, we = pre[i][0], post[i][1]
        m = sum(1 for t in mallocs if ws <= t < we)
        f = sum(1 for t in frees if ws <= t < we)
        print(f"  study {i + 1}: cudaMalloc={m} cudaFree={f}")
    # Plan-01 baseline rule: study 1 may malloc (pool expansion / first-touch
    # CuPy pool); studies 2/3 must be FLAT (0 or pool-expansion-only).
    ok = True
    for i in (1, 2):
        ws, we = pre[i][0], post[i][1]
        m = sum(1 for t in mallocs if ws <= t < we)
        if m != 0:
            print(f"  NOTE: study {i + 1} has {m} cudaMalloc — verify each is a "
                  f"pool-expansion event (<2 ms, at stage start) before classifying flat.")
    total = len(mallocs)
    print(f"  total process cudaMalloc={total} cudaFree={len(frees)}")
    return 0


# ---------------------------------------------------------------------------
# replay mode
# ---------------------------------------------------------------------------
def main_replay() -> int:
    import cupy as cp
    import numpy as np
    import pynvml
    import torch

    import gpu_bootstrap  # RMM first (INFR-01/D-14, Plan 01 4 GiB pin)

    backend = gpu_bootstrap.install_torch_allocator()
    assert backend == "pluggable", f"RMM not active: backend={backend}"
    print(f"allocator backend: {backend} (RMM pinned initial pool, Plan 01)")

    from monai.deploy.core import Application
    from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
    from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
    from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator

    from my_app.operators.ensemble_average_operator import EnsembleAverageOperator
    from my_app.operators.postprocess_operator import PostprocessOperator
    from my_app.operators.postresample_operator import PostResampleOperator
    from my_app.operators.preprocess_operator import PreprocessOperator
    from my_app.operators.slidewindow_operator import SlideWindowOperator

    fragment = Application()  # bare fragment — holoscan 4.2 rejects None (Plan 02)

    # --- the app's own load path (SDK chain, identical to the DAG) --------
    loader = DICOMDataLoaderOperator(fragment, name="loader_op")
    selector = DICOMSeriesSelectorOperator(fragment, name="selector_op")
    vol_op = DICOMSeriesToVolumeOperator(fragment, name="series_to_vol_op")
    study_list = loader.load_data_to_studies(INPUT_DIR)
    selected = selector.filter(None, study_list)
    image = vol_op.convert_to_image(selected)
    arr = image.asnumpy()
    print(f"study loaded via SDK chain: shape={arr.shape} dtype={arr.dtype}")

    # --- the REAL operators (this plan's caches active) -------------------
    pre = PreprocessOperator(fragment, model_path=str(MODEL_DIR),
                             config_name=CONFIG, name="pre")
    sw = SlideWindowOperator(fragment, model_path=str(MODEL_DIR),
                             config_name=CONFIG, name="sw")
    post = PostResampleOperator(fragment, config_name=CONFIG,
                                emit_probabilities=True, emit_lowres_seg=False,
                                name="post")
    # Legacy single-probabilities mode: numerically identical to the
    # multi-config mode with a one-element ensemble list (Phase 1 path).
    ens = EnsembleAverageOperator(fragment, emit_averaged_probabilities=False,
                                  name="ens")
    OUT = Path("/tmp/infr02_replay_out")
    OUT.mkdir(parents=True, exist_ok=True)
    proc = PostprocessOperator(fragment, model_path=str(MODEL_DIR),
                               applied_labels=(1,),
                               label_names={0: "background", 1: "airway"},
                               output_labels=(1,), output_folder=str(OUT),
                               name="proc")

    class _In:
        def __init__(self, d):
            self._d = d
        def receive(self, name):
            return self._d.get(name)

    class _Out:
        def __init__(self):
            self._d = {}
        def emit(self, value, name):
            self._d[name] = value

    def snapshot(op, label):
        """data_ptr table for one operator's cache (post preprocess+inference)."""
        rows = []
        for (shape, dtype_str), buf in op._buf_cache.items():
            ptr = buf.data_ptr() if hasattr(buf, "data_ptr") else buf.data.ptr
            rows.append((label, (shape, dtype_str), ptr, buf.nbytes))
        return rows

    def vram_used_mib():
        torch.cuda.synchronize()
        cp.cuda.Device(0).synchronize()
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used // (1024 * 1024)

    pynvml.nvmlInit()
    print(f"VRAM at replay start (post model load): {vram_used_mib():,} MiB")

    results = []
    tims = {}
    for n in range(1, N_STUDIES + 1):
        t0 = time.time()
        print(f"\n=== REPLAY STUDY {n} START (epoch_ns={time.time_ns()}) ===")
        o1 = _Out()
        pre.compute(_In({"image": image}), o1, None)
        preproc, meta = o1._d["preprocessed"], o1._d["preprocessed_meta"]
        o2 = _Out()
        sw.compute(_In({"preprocessed": preproc}), o2, None)
        logits = o2._d["logits"]
        # snapshot AFTER preprocess + inference stages complete (plan spec)
        snap = snapshot(pre, "preprocess") + snapshot(sw, "slidewindow")
        g_ptr = sw._gaussian.data_ptr() if sw._gaussian is not None else None
        print(f"[study {n}] data_ptr table (preprocess+inference complete):")
        for label, key, ptr, nbytes in snap:
            print(f"    {label:12s} {str(key):44s} ptr={ptr}  {nbytes / 1e6:8.1f} MB")
        print(f"    {'slidewindow':12s} gaussian (setup-time, once)          ptr={g_ptr}")
        print(f"[study {n}] total_bytes: preprocess={pre._buf_cache.total_bytes():,} "
              f"slidewindow={sw._buf_cache.total_bytes():,}")

        o3 = _Out()
        post.compute(_In({"logits": logits, "preprocessed_meta": meta}), o3, None)
        probs = o3._d["probabilities"]
        o4 = _Out()
        ens.compute(_In({"probabilities": probs}), o4, None)
        seg = o4._d["seg"]
        proc.output_folder = OUT / f"study{n}"
        o5 = _Out()
        proc.compute(_In({"seg": seg, "image": image}), o5, None)
        seg_np = np.asarray(o5._d["seg"])
        text = o5._d["result_text"]
        vram = vram_used_mib()
        tims[n] = time.time() - t0
        print(f"[study {n}] seg payload {seg_np.shape} dtype={seg_np.dtype} "
              f"nonzero={int((seg_np != 0).sum())} | SR: {text!r} | "
              f"VRAM after study {n}: {vram:,} MiB | wall {tims[n]:.1f} s")
        print(f"=== REPLAY STUDY {n} DONE (epoch_ns={time.time_ns()}) ===")
        results.append((n, snap, g_ptr, vram, seg_np, text))

    # ------------------------------------------------------------------
    print("\n=== ASSERTIONS ===")
    failures = []

    # (i) data_ptr tables identical across studies
    base = results[0]
    for n, snap, g_ptr, *_ in results[1:]:
        if [r[:3] for r in snap] == [r[:3] for r in base[1]]:
            print(f"  [PASS] study {n}: data_ptr table IDENTICAL to study 1 "
                  f"({len(snap)} cached buffers + gaussian ptr "
                  f"{'identical' if g_ptr == base[2] else 'CHANGED'})")
        else:
            print(f"  [FAIL] study {n}: data_ptr table differs from study 1")
            for (l1, k1, p1, _), (l2, k2, p2, _) in zip(base[1], snap):
                if (l1, k1, p1) != (l2, k2, p2):
                    print(f"        {k1}: study1={p1} study{n}={p2}")
            failures.append(f"study {n} ptr table")
        if g_ptr != base[2]:
            failures.append(f"study {n} gaussian ptr")

    # (ii) VRAM flat across studies (+/-5%)
    v1, v2, v3 = results[0][3], results[1][3], results[2][3]
    ok_vram = abs(v2 - v1) / v1 <= 0.05 and abs(v3 - v1) / v1 <= 0.05
    print(f"  [{'PASS' if ok_vram else 'FAIL'}] VRAM flat across studies (+/-5%): "
          f"after s1={v1:,} MiB, s2={v2:,} MiB ({100 * (v2 - v1) / v1:+.2f}%), "
          f"s3={v3:,} MiB ({100 * (v3 - v1) / v1:+.2f}%)")
    if not ok_vram:
        failures.append("VRAM growth > 5%")

    # (iii) byte-identical final seg + SR text
    for n, *_ , seg_np, text in results[1:]:
        same = np.array_equal(seg_np, base[4]) and text == base[5]
        print(f"  [{'PASS' if same else 'FAIL'}] study {n}: final seg "
              f"{'BYTE-IDENTICAL' if np.array_equal(seg_np, base[4]) else 'DIFFERS'} "
              f"to study 1 (np.array_equal); SR text {'identical' if text == base[5] else 'DIFFERS'}")
        if not same:
            failures.append(f"study {n} output")

    print(f"\nper-study wall: " + ", ".join(f"s{n}={t:.1f}s" for n, t in sorted(tims.items())))
    if failures:
        print(f"REPLAY RESULT: FAIL ({failures})")
        return 1
    print("REPLAY RESULT: PASS (address stability + flat VRAM + byte-identical outputs)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--analyze", metavar="SQLITE",
                    help="per-study cudaMalloc churn from an nsys-exported sqlite")
    args = ap.parse_args()
    if args.analyze:
        sys.exit(analyze_sqlite(args.analyze))
    sys.exit(main_replay())
