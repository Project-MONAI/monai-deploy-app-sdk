#!/usr/bin/env python
"""Headless unit suite for the MEM-003/D-23 weight release hook.

Covers — without a full app run:

  (a) ``SlideWindowOperator.release()`` on a small synthetic bundle
      (2x1x1 random CUDA tensors) -> ``_bundle is None`` afterwards, the
      ``weights released: ... (MEM-003)`` log line is present, and a
      second ``release()`` is a no-op (no exception).
  (b) ``release()`` on the REAL SlideWindowOperator built from the airway
      bundle's 3d_lowres config after ``setup()``: driver-level memory
      (pynvml) sampled before/after; ``_bundle is None`` afterwards and
      ``compute()`` now raises the post-release guard RuntimeError.
  (c) Callback semantics: a counter callable passed as ``release_fn`` to a
      bare PostResampleOperator fires EXACTLY ONCE when the real
      ``compute()`` tail executes (driven with a tiny synthetic logits
      tensor + fake op_input/op_output — no app run).

Measurement discipline (RESEARCH Pitfall 2): torch's CUDA memory counters
(memory stats / allocated / reserved) RAISE under the RMM pluggable
allocator — driver-level memory is sampled with pynvml only. The test runs
with the RMM allocator installed (gpu_bootstrap first, shipping import
order) so ``release()``/``empty_cache()`` are exercised under RMM.

Device pinning (Pitfall 7): CUDA is pinned to device 0 before any torch
CUDA use; the pynvml sample uses the same physical device.

Run:  cd examples/apps/cchmc-nnunet-fast && ulimit -s unlimited && \
      /tmp/monai-env/.venv/bin/python scripts/test_weight_release.py
"""

import logging
import os
import resource
import sys
from pathlib import Path

# Pitfall 7: pin the device BEFORE any torch CUDA context init.
DEVICE_INDEX = int(os.environ.get("MEM003_TEST_DEVICE", "0"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(DEVICE_INDEX))

# Holoscan needs a >= 32 MB stack (the app relies on `ulimit -s unlimited`).
try:
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
except (ValueError, OSError):
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent
MY_APP = APP_ROOT / "my_app"
REPO_ROOT = APP_ROOT.parents[2]
MODEL_PATH = Path(
    os.environ.get(
        "MEM003_TEST_MODEL_PATH",
        str(REPO_ROOT / "examples" / "apps" / "cchmc_nnunet_fifteen_ckpt_app" / "models"),
    )
)

for p in (str(MY_APP / "operators"), str(MY_APP)):
    sys.path.insert(0, p)
sys.path.insert(0, str(APP_ROOT))  # the local my_app package wins resolution

# Shipping import order: RMM allocator before any other heavy import (INFR-01).
import gpu_bootstrap

gpu_bootstrap.install_torch_allocator()

import pynvml  # noqa: E402
import torch  # noqa: E402

from monai.deploy.core import Application  # noqa: E402

from preprocess_operator import to_holoscan_gpu_tensor  # noqa: E402
from postresample_operator import PostResampleOperator  # noqa: E402
from slidewindow_operator import ModelBundle, SlideWindowOperator  # noqa: E402


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


class _TinyNet(torch.nn.Module):
    """A 2x1x1 random-CUDA-tensor stand-in network (case a)."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, 1, 1, device="cuda"))


def _synthetic_bundle() -> ModelBundle:
    return ModelBundle(
        config_name="3d_lowres",
        trainer_name="synthetic",
        network=_TinyNet(),
        fold_state_dicts=[{"w": torch.randn(2, 1, 1, device="cuda")} for _ in range(2)],
        mirror_axes=None,
        patch_size=(1, 1, 1),
        num_segmentation_heads=2,
        use_mirroring=False,
        use_gaussian=False,
        tile_step_size=0.5,
        device=torch.device("cuda"),
    )


def _driver_used_mib(handle) -> int:
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used // (1024 * 1024)


def case_a_release_synthetic(op: SlideWindowOperator, real_bundle, cap: _LogCapture):
    """release() on a synthetic bundle: _bundle None, log line, 2nd no-op."""
    print("== case (a): release() on a synthetic bundle ==")
    ok = True
    cap.records.clear()
    stashed_real = op._bundle
    op._bundle = _synthetic_bundle()
    op._released = False
    op.release()
    ok &= op._bundle is None
    log_lines = [r for r in cap.records if r.startswith("weights released:")]
    ok &= bool(log_lines)
    ok &= any("(MEM-003)" in r and "(folds=2)" in r for r in log_lines)
    try:
        op.release()  # second release must be a no-op
    except Exception as exc:  # noqa: BLE001
        print(f"  second release() raised: {exc!r}")
        ok = False
    print(f"  _bundle is None after release: {op._bundle is None}")
    print(f"  log line: {log_lines[0] if log_lines else None}")
    print("  second release no-op: ok")
    op._bundle = stashed_real  # restore the real bundle for case (b)
    print(f"case (a) {'PASS' if ok else 'FAIL'}")
    return ok


def case_b_release_real(op: SlideWindowOperator, handle):
    """release() on the real 3d_lowres bundle: pynvml before/after + guard."""
    print("== case (b): release() on the real 3d_lowres SlideWindowOperator ==")
    ok = True
    assert op._bundle is not None, "real bundle must be restored before case (b)"
    n_folds = len(op._bundle.fold_state_dicts)
    before = _driver_used_mib(handle)
    op.release()
    after = _driver_used_mib(handle)
    ok &= op._bundle is None
    ok &= n_folds == 5  # the 3d_lowres 5-fold ensemble
    print(
        f"  driver used VRAM (device {DEVICE_INDEX}): {before} MiB before -> "
        f"{after} MiB after release ({after - before:+d} MiB; "
        "RMM pool semantics — see Open Q2, measured at pool level in Task 2)"
    )
    try:
        op.compute(None, None, None)
        print("  compute() after release did NOT raise")
        ok = False
    except RuntimeError as exc:
        guarded = "DAG ordering violation" in str(exc) and "3d_lowres" in str(exc)
        print(f"  compute() after release raised: {exc}")
        ok &= guarded
    except Exception as exc:  # noqa: BLE001
        print(f"  compute() raised wrong type: {exc!r}")
        ok = False
    print(f"case (b) {'PASS' if ok else 'FAIL'}")
    return ok


class _FakeOpInput:
    def __init__(self, holo_logits, meta):
        self._holo_logits = holo_logits
        self._meta = meta

    def receive(self, port):
        if port == PostResampleOperator.INPUT_LOGITS:
            return self._holo_logits
        if port == PostResampleOperator.INPUT_META:
            return self._meta
        return None


class _FakeOpOutput:
    def __init__(self):
        self.emits = []  # (tensor, port)

    def emit(self, tensor, port):
        self.emits.append((tensor, port))


def case_c_release_fn_fires_once():
    """release_fn fires exactly once when the real compute() tail executes."""
    print("== case (c): release_fn callback fires exactly once at the compute() tail ==")
    ok = True
    calls = {"n": 0}

    def release_fn():
        calls["n"] += 1

    app = Application()
    op = PostResampleOperator(
        app,
        emit_lowres_seg=True,
        emit_probabilities=False,
        config_name="3d_lowres",
        release_fn=release_fn,
        name="postresample_test",
    )

    s = 8
    logits = torch.randn(2, s, s, s, dtype=torch.float32, device="cuda")
    meta = {
        "shape_after_cropping_and_before_resampling": [s, s, s],  # == logits -> resample short-circuit
        "shape_before_cropping": [s, s, s],
        "bbox_used_for_cropping": [[0, s], [0, s], [0, s]],
        "transpose_forward": [2, 1, 0],
        "target_spacing": [1.0, 1.0, 1.0],
        "original_spacing": [1.0, 1.0, 1.0],
    }
    fake_in = _FakeOpInput(to_holoscan_gpu_tensor(logits), meta)
    fake_out = _FakeOpOutput()
    try:
        op.compute(fake_in, fake_out, None)
    except Exception as exc:  # noqa: BLE001
        print(f"  compute() raised: {exc!r}")
        return False

    lowres_emitted = any(port == PostResampleOperator.OUTPUT_LOWRES_SEG for _, port in fake_out.emits)
    ok &= lowres_emitted
    ok &= calls["n"] == 1
    print(f"  lowres_seg emitted: {lowres_emitted}; release_fn calls after 1 compute: {calls['n']}")

    # release_fn=None keeps byte-for-byte prior behavior (no callback).
    op_none = PostResampleOperator(
        app,
        emit_lowres_seg=False,
        emit_probabilities=False,
        config_name="3d_fullres",
        name="postresample_test_none",
    )
    assert op_none._release_fn is None
    print(f"case (c) {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(DEVICE_INDEX)
    assert torch.cuda.memory.get_allocator_backend() == "pluggable", (
        "test must run under the RMM pluggable allocator (shipping import order)"
    )

    cap = _LogCapture()
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(cap)

    print(f"building real SlideWindowOperator (model_path={MODEL_PATH}, config=3d_lowres) ...")
    app = Application()
    op = SlideWindowOperator(
        app, model_path=str(MODEL_PATH), config_name="3d_lowres", name="slidewindow_test"
    )
    assert op._bundle is not None and op.model_load_count == 1, "setup() must load the real bundle"

    failures = []
    failures.append(not case_a_release_synthetic(op, op._bundle, cap))
    failures.append(not case_b_release_real(op, handle))
    failures.append(not case_c_release_fn_fires_once())

    if any(failures):
        print("RESULT: FAIL")
        sys.exit(1)
    print("RESULT: PASS (release hook semantics verified headlessly under RMM)")


if __name__ == "__main__":
    main()
