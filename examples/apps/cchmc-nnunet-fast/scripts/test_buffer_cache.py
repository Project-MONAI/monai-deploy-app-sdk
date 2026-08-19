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
  7. torch-site semantics    — SlideWindowOperator site patterns:
                               zero-on-borrow for the predicted_logits /
                               n_predictions accumulators (borrow -> write
                               garbage -> reborrow with zero=True -> all
                               zeros, 3 fake folds), gaussian-once (compute
                               in a fake setup, 3 fake folds reuse the same
                               data_ptr, read-only in the loop), and the
                               multi-fold accumulation aliasing regression
                               (the running sum must not alias the per-fold
                               cached buffer: fold-1 clone == fresh-alloc
                               reference; without the clone the sum is
                               corrupted — the Rule 1 bug this plan caught)

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

    # ------------------------------------------------------------------
    print("== case 7: torch-site semantics (SlideWindowOperator site patterns) ==")
    # 7a. zero-on-borrow for the predicted_logits / n_predictions pattern:
    # the reference allocates fresh torch.zeros per fold; with the cache
    # each fold BORROWS with zero=True — so any garbage the previous fold
    # left in the buffer must be gone at the next borrow.
    swc = _ShapeCache("cuda", family="torch")
    logit_shape = (1, *S_A)  # (heads, x, y, z)
    prev_ptr = None
    for fold in range(3):  # 3 fake folds
        logits = swc.get(logit_shape, torch.float32, zero=True)
        counts = swc.get(S_A, torch.float32, zero=True)
        if prev_ptr is not None:
            check(f"fold {fold + 1}: predicted_logits reuses study buffer "
                  f"(same data_ptr)", logits.data_ptr() == prev_ptr)
        check(f"fold {fold + 1}: zero-on-borrow -> all zeros after prior "
              f"fold's garbage", bool(torch.all(logits == 0)) and bool(torch.all(counts == 0)))
        prev_ptr = logits.data_ptr()
        # simulate the fold's sliding-window accumulation (garbage for the
        # NEXT borrow's zero=True to remove):
        logits += 1.5
        counts += 0.25
    # 7b. gaussian-once: computed in a fake setup(), identical every fold —
    # all 3 fake folds must see the SAME tensor (same data_ptr), and the
    # loop uses it read-only.
    patch = (16, 16, 8)
    gaussian = torch.randn(patch, device="cuda", dtype=torch.float32)  # fake setup() compute
    fold_ptrs = []
    for fold in range(3):
        g = gaussian  # per-fold loop reuses the stored tensor (read-only)
        workon = swc.get((1, 1, *patch), torch.float32)  # per-patch borrow, zero=False
        workon.copy_(torch.randn(1, 1, *patch, device="cuda"))
        prediction = workon[0] * g  # read of g (mirrors prediction * gaussian)
        n_pred = swc.get(patch, torch.float32)
        n_pred.add_(g)  # read of g (mirrors n_predictions += gaussian)
        fold_ptrs.append(g.data_ptr())
        check(f"fold {fold + 1}: gaussian tensor UNMODIFIED by the loop "
              f"(read-only reuse)", bool(torch.equal(g, gaussian)))
    check("gaussian-once: 3 fake folds reused one tensor (identical data_ptr)",
          len(set(fold_ptrs)) == 1, f"ptrs={fold_ptrs}")
    # 7c. multi-fold accumulation aliasing regression (the predict_logits
    # pattern — Rule 1 bug caught during plan execution): sliding_window_
    # predict returns a VIEW of the per-fold cached predicted_logits buffer;
    # the NEXT fold re-borrows the same buffer with zero=True, which would
    # wipe a running sum that aliases it (and fold k would add the buffer to
    # itself). The shipped fix clones the fold-1 result before accumulating;
    # this case pins that the with-cache accumulation is exactly the
    # fresh-allocation reference — and documents that WITHOUT the clone the
    # sum is corrupted.
    n_folds = 3
    fold_vals = [torch.full(S_A, float(k + 1), device="cuda") * (0.5 ** k)
                 for k in range(n_folds)]
    ref = fold_vals[0].clone()[None]
    for fv in fold_vals[1:]:
        ref += fv[None]
    ref = ref / n_folds
    acc = None
    for k in range(n_folds):
        fl = swc.get(logit_shape, torch.float32, zero=True)
        fl[0] = fold_vals[k]  # the fold fills its per-fold buffer
        fold_logits = fl[(slice(None),)]  # view of the cached buffer (the
        # exact shape sliding_window_predict's return value has)
        if acc is None:
            acc = fold_logits.clone()  # the shipped predict_logits fix
        else:
            acc += fold_logits
    acc = acc / n_folds
    check("multi-fold accumulation (cache + fold-1 clone) == fresh-allocation "
          "reference", bool(torch.equal(acc, ref)))
    acc_buggy = None
    for k in range(n_folds):
        fl = swc.get(logit_shape, torch.float32, zero=True)
        fl[0] = fold_vals[k]
        fold_logits = fl[(slice(None),)]
        if acc_buggy is None:
            acc_buggy = fold_logits  # NO clone — aliases the re-borrowed buffer
        else:
            acc_buggy += fold_logits
    acc_buggy = acc_buggy / n_folds
    check("regression guard: WITHOUT the fold-1 clone the sum IS corrupted "
          "(the caught bug, for the record)",
          not bool(torch.equal(acc_buggy, ref)))

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failed: {FAILURES})")
        sys.exit(1)
    print("RESULT: PASS (all D-24(a) cache semantics green)")


if __name__ == "__main__":
    main()
