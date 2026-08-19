# Phase 3 live probes (2026-08-19, A100-SXM4-40GB, venv /tmp/monai-env/.venv)

Run with a 32 MB stack and a free GPU pinned:
    ulimit -s 32768; CUDA_VISIBLE_DEVICES=<free> /tmp/monai-env/.venv/bin/python <probe>.py [args]

- `probe_concurrency.py <cpu|gpu> <default|multithread|eventbased>` — does the
  4.2 scheduler run independent DAG branches concurrently? (result: default
  GreedyScheduler serial 5.30 s; MultiThread/EventBased concurrent 3.40 s,
  distinct thread IDs, overlapping spans)
- `probe_gpu_overlap.py <defstreams|ownstreams>` — GPU-level overlap of two
  saturating torch branches under EventBasedScheduler (result: ~10 % at best;
  own-streams variant no gain)
- `probe3.py` — 3-way concurrency + CountCondition(3) join fires exactly once
  (wall 2.13 s vs 4.5 s serial)
- `probe_resample_identity.py` — scipy 1.15.3 vs cupy 14.1.1 byte-identity for
  map_coordinates/zoom at orders 0/1/3, fp32/fp64 (result: o0 byte-equal;
  o1/o3 not byte-equal even in fp64 — accumulation order)

Full findings: .planning/phases/03-optimization/03-RESEARCH.md
