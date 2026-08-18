#!/usr/bin/env python
"""gpu_residency.py — GPU-residency test for the cchmc-nnunet-fast operator
chain (TEST-004).

Verifies the Phase 1 GPU-residency contract: every intermediate tensor stays
on CUDA at every operator boundary, and the ONLY GPU->CPU transfers in the
chain are the two documented, by-design ones:

  1. ``postresample_operator.py`` — the reference-parity CPU (scipy)
     resample path. Phase 0/1 decision: resampling stays on the reference
     CPU path for pixel-exactness (logits go CPU, are resampled, and return
     to CUDA within the same operator).
  2. ``postprocess_operator.py`` — the EXACTLY-ONCE boundary transfer
     ``seg_gpu.cpu().numpy()`` at the final output stage (the single
     permitted transfer before the DICOM-SEG write).

``sc_overlay.py`` is a third, non-transfer site: the SC overlay math runs on
the ALREADY-TRANSFERRED CPU numpy seg (final output stage); any
``.cpu()``/``.numpy()`` it performs operates on CPU data, never on a
pipeline tensor, so it is allow-listed as such.

Any other ``.cpu()`` / ``.numpy()`` call originating in the app's own
code is a premature CPU transfer and fails the test.

Two detection layers:

``--static`` (no GPU needed, CI-friendly)
  AST-scan the five operator files for ``.cpu()`` / ``.numpy()`` calls,
  check each against the allow-list above, verify the postprocess boundary
  transfer occurs exactly once, and verify every operator's ``compute()``
  carries an ``assert_on_gpu`` entry/exit device guard.

``--runtime`` (needs GPU + bundle + airway DICOM)
  Install hooks on ``torch.Tensor.cpu`` / ``torch.Tensor.numpy`` that
  attribute every call to the innermost frame in the app's ``my_app/``
  code, run the full app end-to-end on the airway study (3d_fullres),
  then check: no unexpected transfers, exactly one postprocess boundary
  transfer, the resample path was exercised, and the SEG was written.
  The per-operator ``assert_on_gpu`` guards execute at every boundary
  during the run and raise if any boundary tensor is off-device.

``--self-test``
  Like ``--runtime``, but first injects an illegal ``.cpu()`` call into an
  intermediate operator (SlideWindow's guard is wrapped to call ``t.cpu()``)
  and verifies the detector flags it — proving the test actually fails on
  an injected violation.

Usage (from the app root, venv python, with ``ulimit -s unlimited``):
  python scripts/gpu_residency.py [--static] [--runtime] [--self-test]
      [--input <dicom dir>] [--model <bundle dir>] [--output <out dir>]

With no mode flag, both the static scan and the runtime E2E check run.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent
REPO_ROOT = APP_ROOT.parents[2]

# The five Phase-1 GPU operators (the pipeline's operator chain).
OPERATOR_FILES = [
    "preprocess_operator.py",
    "slidewindow_operator.py",
    "postresample_operator.py",
    "ensemble_average_operator.py",
    "postprocess_operator.py",
]

# Every app module whose .cpu()/.numpy() call sites the static scan must
# classify (operators + the post-boundary SC overlay helpers).
SCAN_FILES = OPERATOR_FILES + ["sc_overlay.py"]

# Files allowed to perform GPU->CPU transfers, and why.
ALLOWED_TRANSFER_FILES = {
    "postresample_operator.py": "reference-parity CPU (scipy) resample path "
                                "(Phase 0/1 decision; returns to CUDA in-operator)",
    "postprocess_operator.py": "EXACTLY-ONCE boundary transfer seg_gpu.cpu().numpy() "
                               "at the final output stage",
    "sc_overlay.py": "final output stage: SC overlay math on the ALREADY-TRANSFERRED "
                     "CPU numpy seg (runs after the boundary transfer, never on a "
                     "pipeline tensor)",
}

TRANSFER_ATTRS = ("cpu", "numpy")


# --------------------------------------------------------------------------
# static AST scan
# --------------------------------------------------------------------------

def _find_transfers(tree: ast.AST):
    """Yield (lineno, funcname, attr) for every <expr>.cpu()/.numpy() call."""
    results = []

    def visit(node: ast.AST, funcname: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
            else:
                if (isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and child.func.attr in TRANSFER_ATTRS):
                    results.append((child.lineno, funcname, child.func.attr))
                visit(child, funcname)

    visit(tree, "<module>")
    return results


def _compute_has_gpu_guard(tree: ast.AST) -> bool:
    """True if some compute() method in the file calls assert_on_gpu."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "compute":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    f = sub.func
                    if (isinstance(f, ast.Name) and f.id == "assert_on_gpu") or \
                       (isinstance(f, ast.Attribute) and f.attr == "assert_on_gpu"):
                        return True
    return False


def run_static(app_root: Path) -> int:
    ops_dir = app_root / "my_app" / "operators"
    print("=" * 72)
    print("gpu_residency: STATIC scan of the five GPU operators")
    print("=" * 72)
    violations = []
    n_calls = 0
    for fname in SCAN_FILES:
        path = ops_dir / fname
        if not path.is_file():
            violations.append(f"{fname}: file not found")
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        calls = _find_transfers(tree)
        n_calls += len(calls)
        allowed = fname in ALLOWED_TRANSFER_FILES
        status = "ALLOWED" if (allowed or not calls) else "VIOLATION"
        print(f"  {fname}: {len(calls)} .cpu()/.numpy() call(s)  [{status}]")
        for lineno, funcname, attr in calls:
            mark = "ok" if allowed else "VIOLATION"
            print(f"      line {lineno}  .{attr}() in {funcname}  ->  {mark}")
            if not allowed:
                violations.append(
                    f"{fname}:{lineno}: .{attr}() call in {funcname} is not "
                    f"an allowed transfer site")
        # assert_on_gpu guard check applies to the five operators only
        if fname in OPERATOR_FILES:
            if not _compute_has_gpu_guard(tree):
                violations.append(f"{fname}: compute() has no assert_on_gpu device guard")
                print(f"      compute() assert_on_gpu guard: MISSING -> VIOLATION")
            else:
                print(f"      compute() assert_on_gpu guard: present")

    # exactly-once boundary transfer in postprocess
    pp_path = ops_dir / "postprocess_operator.py"
    if pp_path.is_file():
        tree = ast.parse(pp_path.read_text(), filename=str(pp_path))
        pp_cpu = [c for c in _find_transfers(tree) if c[2] == "cpu"]
        if len(pp_cpu) != 1:
            violations.append(
                f"postprocess_operator.py: expected exactly 1 .cpu() boundary "
                f"transfer, found {len(pp_cpu)}")
        print(f"  postprocess boundary .cpu() transfer count: {len(pp_cpu)} "
              f"({'ok' if len(pp_cpu) == 1 else 'VIOLATION: expected 1'})")

    print(f"  transfer call sites scanned: {n_calls}")
    print(f"  RESULT: {'PASS' if not violations else 'FAIL'}")
    for v in violations:
        print(f"    - {v}")
    return 0 if not violations else 1


# --------------------------------------------------------------------------
# runtime instrumentation
# --------------------------------------------------------------------------

class TransferRecorder:
    """Hooks torch.Tensor.cpu/.numpy and records calls attributed to the
    app's own code (innermost frame under app_root/my_app/)."""

    def __init__(self, app_root: Path):
        import torch
        self.torch = torch
        self.app_dir = str(app_root / "my_app")
        self.records = []
        self._orig_cpu = torch.Tensor.cpu
        self._orig_numpy = torch.Tensor.numpy
        rec = self

        def _hook_cpu(tensor, *args, **kwargs):
            rec._record("cpu", tensor)
            return rec._orig_cpu(tensor, *args, **kwargs)

        def _hook_numpy(tensor, *args, **kwargs):
            rec._record("numpy", tensor)
            return rec._orig_numpy(tensor, *args, **kwargs)

        torch.Tensor.cpu = _hook_cpu
        torch.Tensor.numpy = _hook_numpy

    def _record(self, method: str, tensor) -> None:
        # frame 0 = _record, 1 = hook, 2 = the caller of .cpu()/.numpy()
        frame = sys._getframe(2)
        while frame is not None:
            fname = frame.f_code.co_filename
            if fname.startswith(self.app_dir + "/") or fname.startswith(self.app_dir + "\\"):
                dev = str(tensor.device)
                self.records.append({
                    "method": method,
                    "device": dev,
                    "file": Path(fname).name,
                    "line": frame.f_lineno,
                    "func": frame.f_code.co_name,
                })
                return
            frame = frame.f_back
        # library-internal transfer — not attributable to app code

    def uninstall(self):
        self.torch.Tensor.cpu = self._orig_cpu
        self.torch.Tensor.numpy = self._orig_numpy

    def unexpected(self):
        return [r for r in self.records if r["file"] not in ALLOWED_TRANSFER_FILES]

    def boundary(self):
        return [r for r in self.records
                if r["file"] == "postprocess_operator.py" and r["method"] == "cpu"]

    def resample(self):
        return [r for r in self.records if r["file"] == "postresample_operator.py"]


def _run_app_e2e(app_root: Path, input_dir: Path, model_dir: Path,
                 output_dir: Path) -> None:
    """Run the assembled app in-process (3d_fullres, single config)."""
    for p in (str(app_root / "my_app"), str(app_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    sys.argv = [sys.executable, "-i", str(input_dir), "-m", str(model_dir),
                "-o", str(output_dir)]
    fast_app = importlib.import_module("my_app.app")
    print(f"  running E2E: input={input_dir} model={model_dir} out={output_dir}")
    fast_app.CCHMCNNUnetFastApp().run()


def _check_runtime(recorder: TransferRecorder, output_dir: Path):
    """Evaluate the recorded transfers. Returns (ok, messages)."""
    msgs = []
    unexpected = recorder.unexpected()
    boundary = recorder.boundary()
    resample = recorder.resample()
    seg_files = list((output_dir / "SEG").glob("*.dcm"))

    print(f"  app-attributed .cpu()/.numpy() calls: {len(recorder.records)}")
    for r in recorder.records:
        mark = "ok" if r["file"] in ALLOWED_TRANSFER_FILES else "VIOLATION"
        print(f"      .{r['method']}() {r['file']}:{r['line']} in {r['func']} "
              f"(from {r['device']})  ->  {mark}")
    print(f"  resample-path transfers: {len(resample)} (allowed, by design)")
    print(f"  postprocess boundary .cpu() transfers: {len(boundary)} (must be 1)")
    print(f"  SEG written: {len(seg_files)} file(s)")

    if not recorder.records:
        msgs.append("no app-attributed transfers recorded — the app chain did "
                    "not run (or the hooks were bypassed)")
    if unexpected:
        for r in unexpected:
            msgs.append(f"UNEXPECTED CPU transfer: .{r['method']}() at "
                        f"{r['file']}:{r['line']} in {r['func']} "
                        f"(premature transfer before the final stage)")
    if len(boundary) != 1:
        msgs.append(f"expected exactly 1 postprocess boundary .cpu() transfer, "
                    f"saw {len(boundary)}")
    if not resample:
        msgs.append("no resample-path transfer recorded (postresample CPU path "
                    "was not exercised)")
    if not seg_files:
        msgs.append("no DICOM-SEG written to the output dir")
    return (not msgs), msgs


def run_runtime(app_root: Path, input_dir: Path, model_dir: Path,
                output_dir: Path) -> int:
    print("=" * 72)
    print("gpu_residency: RUNTIME E2E (3d_fullres, airway study)")
    print("=" * 72)
    recorder = TransferRecorder(app_root)
    try:
        try:
            _run_app_e2e(app_root, input_dir, model_dir, output_dir)
        except RuntimeError as e:
            # assert_on_gpu / assert_cuda_available guards raise RuntimeError
            print(f"  RUNTIME VIOLATION (operator guard raised): {e}")
            return 1
        ok, msgs = _check_runtime(recorder, output_dir)
    finally:
        recorder.uninstall()
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    for m in msgs:
        print(f"    - {m}")
    return 0 if ok else 1


def run_self_test(app_root: Path, input_dir: Path, model_dir: Path,
                  output_dir: Path) -> int:
    print("=" * 72)
    print("gpu_residency: SELF-TEST (injected illegal .cpu() in SlideWindow)")
    print("=" * 72)
    # Ensure the app packages are importable, then inject the violation.
    for p in (str(app_root / "my_app"), str(app_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    sw = importlib.import_module("my_app.operators.slidewindow_operator")
    orig_guard = sw.assert_on_gpu

    def injected_guard(tensor):
        # injected ILLEGAL intermediate CPU transfer (self-test only)
        _ = tensor.cpu()
        return orig_guard(tensor)

    sw.assert_on_gpu = injected_guard
    print("  injected .cpu() call into slidewindow assert_on_gpu wrapper")
    recorder = TransferRecorder(app_root)
    rc = 1  # default: self-test fails unless detection + correct verdict
    try:
        try:
            _run_app_e2e(app_root, input_dir, model_dir, output_dir)
            ok, msgs = _check_runtime(recorder, output_dir)
        except RuntimeError as e:
            print(f"  (run aborted by guard: {e})")
            ok, msgs = False, [str(e)]
        unexpected = recorder.unexpected()
        detected = any(r["file"] == "slidewindow_operator.py"
                       for r in unexpected)
        print(f"  injected violation detected by the detector: {detected}")
        print(f"  runtime check verdict on the injected run: "
              f"{'PASS' if ok else 'FAIL'} (FAIL expected)")
        if detected and not ok:
            print("  RESULT: PASS — the test correctly fails on an injected "
                  "intermediate .cpu() call")
            rc = 0
        else:
            print("  RESULT: FAIL — the detector missed the injected violation "
                  "or did not fail the run")
    finally:
        recorder.uninstall()
        sw.assert_on_gpu = orig_guard
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GPU-residency test")
    ap.add_argument("--static", action="store_true",
                    help="run the AST static scan (no GPU needed)")
    ap.add_argument("--runtime", action="store_true",
                    help="run the instrumented E2E check (GPU needed)")
    ap.add_argument("--self-test", action="store_true",
                    help="inject an illegal .cpu() and verify the detector "
                         "flags it (GPU needed)")
    ap.add_argument("--input", default=str(REPO_ROOT / "testdata" / "airway_input"),
                    help="DICOM study dir (default: testdata/airway_input)")
    ap.add_argument("--model",
                    default=str(REPO_ROOT / "examples" / "apps" /
                                "cchmc_nnunet_fifteen_ckpt_app" / "models"),
                    help="nnUNet bundle dir (default: reference app models)")
    ap.add_argument("--output", default="/tmp/gpu_residency_out",
                    help="E2E output dir (default: /tmp/gpu_residency_out)")
    args = ap.parse_args(argv)

    app_root = APP_ROOT
    none_selected = not (args.static or args.runtime or args.self_test)
    do_static = args.static or none_selected
    do_runtime = args.runtime or none_selected
    do_self = args.self_test

    rc = 0
    if do_static:
        rc |= run_static(app_root)
    if do_self:
        rc |= run_self_test(app_root, Path(args.input), Path(args.model),
                            Path(args.output))
    elif do_runtime:
        rc |= run_runtime(app_root, Path(args.input), Path(args.model),
                          Path(args.output))
    return rc


if __name__ == "__main__":
    sys.exit(main())
