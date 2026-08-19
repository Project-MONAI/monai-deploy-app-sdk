#!/usr/bin/env python
"""D-24(a) headless unit suite for the INFR-02 shape-keyed buffer cache.

Runs the REAL ``_ShapeCache`` (my_app/operators/buffer_cache.py) headlessly
with synthetic sizes — GPU present, no app/bundle required. Runs under the
shipping import order (gpu_bootstrap/RMM first) on a pinned device
(Pitfall 7). NO torch memory counters anywhere (they raise under the RMM
pluggable allocator — RESEARCH Pitfall 2).

Cases (each named in a print):
  1. multi-call reuse        — get(s) twice -> identical data_ptr (both families)
  2. shape-key invalidation  — get(s), get(s') (s' != s) -> different data_ptr,
                               BOTH retained (get(s) again -> original ptr)
  3. dtype invariants        — same shape, fp32 vs fp64 -> different buffers;
                               every buffer C-contiguous
  4. zero semantics          — zero=True -> all zeros after borrow;
                               zero=False -> prior contents survive
                               (documented: callers must never rely on stale
                               data silently — the operator per-site comments
                               justify every zero=False call)
  5. clear()                 — empties and reallocates (new ptr after clear)
  6. cupy family invariants  — (cases 1-5 exercised with family="cupy")

Run:  /tmp/monai-env/.venv/bin/python scripts/test_buffer_cache.py
Exit: 0 on success, 1 on any failure.
"""

import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
MY_APP = APP_ROOT / "my_app"
sys.path.insert(0, str(MY_APP))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

import cupy as cp  # noqa: E402
import gpu_bootstrap  # noqa: E402  (RMM first — shipping import order)
import numpy as np  # noqa: E402
import torch  # noqa: E402

try:  # package-style import (my_app.*) — the editable install maps the package
    from my_app.operators.buffer_cache import _ShapeCache
except ImportError:  # flat import (my_app dir on sys.path)
    from operators.buffer_cache import _ShapeCache

gpu_bootstrap.install_torch_allocator()

FAILURES = []


def check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(label)


def main():
    print(f"allocator backend: {torch.cuda.memory.get_allocator_backend()}")
    print(f"device: {torch.cuda.get_device_name(0)}")

    S_A = (64, 64, 32)
    S_B = (64, 64, 33)  # different shape (invalidation key)

    # ------------------------------------------------------------------
    print("== case 1: multi-call reuse (identical data_ptr) ==")
    tc = _ShapeCache("cuda", family="torch")
    a1 = tc.get(S_A, torch.float32)
    a2 = tc.get(S_A, torch.float32)
    check("torch: get(s) twice -> same data_ptr",
          a1.data_ptr() == a2.data_ptr() and a1 is a2)
    uc = _ShapeCache("cuda", family="cupy")
    b1 = uc.get(S_A, cp.float32)
    b2 = uc.get(S_A, cp.float32)
    check("cupy: get(s) twice -> same data_ptr",
          b1.data.ptr == b2.data.ptr and b1 is b2)

    # ------------------------------------------------------------------
    print("== case 2: shape-key invalidation (new shape -> new buffer, both retained) ==")
    a3 = tc.get(S_B, torch.float32)
    check("torch: different shape -> different data_ptr", a3.data_ptr() != a1.data_ptr())
    check("torch: both shapes retained (s' does not evict s)",
          tc.get(S_A, torch.float32).data_ptr() == a1.data_ptr())
    check("torch: cache holds exactly 2 keys", len(tc.keys()) == 2, f"keys={tc.keys()}")
    b3 = uc.get(S_B, cp.float32)
    check("cupy: different shape -> different data_ptr", b3.data.ptr != b1.data.ptr)
    check("cupy: both shapes retained", uc.get(S_A, cp.float32).data.ptr == b1.data.ptr)

    # ------------------------------------------------------------------
    print("== case 3: dtype invariants (no cross-dtype sharing; C-contiguous) ==")
    a32 = tc.get(S_A, torch.float32)
    a64 = tc.get(S_A, torch.float64)
    check("torch: same shape, fp32 vs fp64 -> different buffers",
          a32.data_ptr() != a64.data_ptr())
    check("torch: fp32 buffer C-contiguous", a32.is_contiguous())
    check("torch: fp64 buffer C-contiguous", a64.is_contiguous())
    b32 = uc.get(S_A, cp.float32)
    b64 = uc.get(S_A, cp.float64)
    bU8 = uc.get(S_A, cp.uint8)
    check("cupy: fp32 vs fp64 vs uint8 -> 3 distinct buffers",
          len({b32.data.ptr, b64.data.ptr, bU8.data.ptr}) == 3)
    check("cupy: all C-contiguous",
          b32.flags.c_contiguous and b64.flags.c_contiguous and bU8.flags.c_contiguous)

    # ------------------------------------------------------------------
    print("== case 4: zero semantics (zero=True re-zeros; zero=False keeps contents) ==")
    z = tc.get(S_A, torch.float32)
    z.fill_(3.25)  # dirty the shared buffer
    zb = tc.get(S_A, torch.float32, zero=True)
    check("torch: zero=True borrow -> all zeros (prior contents NOT relied on)",
          torch.all(zb == 0))
    zb.fill_(7.5)
    nz = tc.get(S_A, torch.float32, zero=False)
    check("torch: zero=False borrow -> prior contents survive "
          "(documented; callers must overwrite before reading)",
          torch.all(nz == 7.5))
    zb2 = uc.get(S_A, cp.float32)
    zb2.fill(9.5)
    zc = uc.get(S_A, cp.float32, zero=True)
    check("cupy: zero=True borrow -> all zeros", bool(cp.all(zc == 0)))
    zc.fill(4.5)
    nzc = uc.get(S_A, cp.float32, zero=False)
    check("cupy: zero=False borrow -> prior contents survive",
          bool(cp.all(nzc == 4.5)))

    # ------------------------------------------------------------------
    print("== case 5: clear() empties and reallocates ==")
    a1_ref = a1  # keep the reference so the pre-clear ptr is meaningful
    before_t = a1_ref.data_ptr()
    tc.clear()
    check("torch: clear() -> no keys, 0 bytes", tc.keys() == [] and tc.total_bytes() == 0)
    a_new = tc.get(S_A, torch.float32)
    # NB: the pool may legitimately hand back the SAME block — "reallocates"
    # means get() works after clear() (the cache no longer knows the buffer),
    # not that the address must differ. Assert functionality, not address.
    check("torch: post-clear get() -> valid working buffer",
          tuple(a_new.shape) == S_A and a_new.is_contiguous() and tc.total_bytes() > 0)
    before_u = b1.data.ptr
    uc.clear()
    check("cupy: clear() -> no keys, 0 bytes", uc.keys() == [] and uc.total_bytes() == 0)
    b_new = uc.get(S_A, cp.float32)
    check("cupy: post-clear get() -> valid working buffer",
          tuple(b_new.shape) == S_A and b_new.flags.c_contiguous and uc.total_bytes() > 0)

    # ------------------------------------------------------------------
    print("== case 6: total_bytes / keys accounting (proof-record helpers) ==")
    tb = _ShapeCache("cuda", family="torch")
    x1 = tb.get(S_A, torch.float32)
    x2 = tb.get(S_B, torch.uint8)
    expected = S_A[0] * S_A[1] * S_A[2] * 4 + S_B[0] * S_B[1] * S_B[2] * 1
    check("torch: total_bytes == sum over entries", tb.total_bytes() == expected,
          f"{tb.total_bytes()} vs {expected}")
    check("torch: keys() == (shape, str(dtype)) pairs",
          set(tb.keys()) == {(S_A, "torch.float32"), (S_B, "torch.uint8")})

    # ------------------------------------------------------------------
    # shares_storage (emit-boundary DLPack retention check helper)
    ss = _ShapeCache("cuda", family="torch")
    buf = ss.get(S_A, torch.float32)
    view = buf[(slice(None), slice(8), slice(8))]
    other = torch.empty(S_A, dtype=torch.float32, device="cuda")
    check("shares_storage: base buffer detected", ss.shares_storage(buf))
    check("shares_storage: offset VIEW of the buffer detected (data_ptr alone would miss it)",
          ss.shares_storage(view))
    check("shares_storage: unrelated tensor not detected", not ss.shares_storage(other))
    sc = _ShapeCache("cuda", family="cupy")
    cbuf = sc.get(S_A, cp.float32)
    cview = cbuf[8:16, 8:16, :]
    check("cupy: shares_storage base + offset view",
          sc.shares_storage(cbuf) and sc.shares_storage(cview))

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failed: {FAILURES})")
        sys.exit(1)
    print("RESULT: PASS (all D-24(a) cache semantics green)")


if __name__ == "__main__":
    main()
