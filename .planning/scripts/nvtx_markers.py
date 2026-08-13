"""
NVTX range-marker helpers for annotating profiling timelines.

Integrates with ``torch.cuda.nvtx`` so that markers show up in
Nsight Systems, Nsight Compute, and any other CUDA profiler.

Usage inside a profiled script:
    from nvtx_markers import push_range, pop_range, range_colors

    with push_range("preprocessing", range_colors.PREPROCESS):
        ...  # your data-load / augmentation code

    with push_range("forward pass", range_colors.INFERENCE):
        output = model(input_tensor)

    with push_range("post-processing", range_colors.POSTPROCESS):
        result = decode(output)
"""

from contextlib import contextmanager
from typing import Optional

# ── Common colours (hex, no '#' prefix) ───────────────────────────────────────
# These map to standard Nsight Systems category colours.
_RANGE_COLORS = {
    # Pipeline stages
    "preprocess":  "66BB6A",   # green
    "inference":   "42A5F5",   # blue
    "postprocess": "FFA726",   # orange

    # Data movement
    "data-load":   "AB47BC",   # purple
    "data-transfer": "26C6DA", # cyan

    # Model internals
    "encoder":     "EC407A",   # pink
    "decoder":     "5C6BC0",   # indigo
    "fusion":      "FFEE58",   # yellow
    "normalization": "8BC34A", # lime

    # Miscellaneous
    "misc":        "BDBDBD",   # grey
    "checkpoint":  "EF5350",   # red
}

# Convenience namedtuple so callers can write ``range_colors.INFERENCE``.
from types import SimpleNamespace
range_colors = SimpleNamespace(**_RANGE_COLORS)


# ── Core context managers ─────────────────────────────────────────────────────

@contextmanager
def push_range(message: str, color: Optional[str] = None):
    """Context manager that opens an NVTX range on enter and closes it on exit.

    Falls back gracefully when PyTorch CUDA is not available (e.g. CPU-only
    machines or import-time checks).

    Parameters
    ----------
    message : str
        Label shown in the profiler timeline.
    color : str, optional
        Hex colour (6 chars, no '#').  Defaults to ``range_colors.INFERENCE``.
    """
    try:
        import torch.cuda.nvtx as nvtx  # type: ignore[attr-defined]
        if color is None:
            color = _RANGE_COLORS["inference"]
        nvtx.range_push(message)
        yield
    except ImportError:
        # torch.cuda.nvtx not available — silently passthrough
        yield
    finally:
        try:
            import torch.cuda.nvtx as nvtx  # type: ignore[attr-defined]
            nvtx.range_pop()
        except ImportError:
            pass


@contextmanager
def pop_range():
    """Explicitly pop the current NVTX range.

    Typically you won't need this directly — use ``push_range`` as a context
    manager instead.  Provided for manual pairing when you need fine-grained
    control.
    """
    try:
        import torch.cuda.nvtx as nvtx  # type: ignore[attr-defined]
        nvtx.range_pop()
    except ImportError:
        pass
    yield


# ── Quick-start: auto-wrap __main__ ───────────────────────────────────────────

def profile_main():
    """Ensure CUDA profiling is started/stopped around ``__main__``.

    Call once at the top of your script:

        if __name__ == "__main__":
            from nvtx_markers import profile_main
            with profile_main():
                main()

    This emits ``cudaProfilerStart`` / ``cudaProfilerStop`` so that
    ``nsys`` (when run with ``--capture-range=cudaProfilerApi``) only
    records the annotated region.
    """
    from contextlib import contextmanager as _ctx

    @_ctx
    def _wrapper():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.cudart().cudaProfilerStart()
        except Exception:
            pass
        yield
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.cudart().cudaProfilerStop()
        except Exception:
            pass

    return _wrapper()


# ── Standalone demo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("NVTX marker helpers loaded.")
    print(f"Available colours: {list(_RANGE_COLORS.keys())}")
    print()
    print("Example usage:")
    print(
        '    with push_range("forward", range_colors.INFERENCE):'
        '\n        model(x)'
    )
