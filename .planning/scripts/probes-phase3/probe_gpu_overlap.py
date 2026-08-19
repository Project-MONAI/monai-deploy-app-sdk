"""Live probe 2: do two torch branches running on operator-allocated CUDA
streams actually overlap at the GPU level (wall-time reduction)?

DAG: source -> workA -> join ; source -> workB -> join
Each WorkOp allocates a holoscan stream (context.allocate_cuda_stream) and
runs SECS of matmuls under torch.cuda.ExternalStream on that stream.
Runs under EventBasedScheduler(worker_thread_number=2).

Variants:
  ownstreams  - each op on its own holoscan-allocated stream
  defstreams  - both ops on the torch default stream (control)
"""
import sys
import threading
import time

SCHED = "eventbased"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "ownstreams"
SECS = 2.0

RECORDS = []


def work_gpu(secs, stream_handle):
    import torch

    torch.cuda.init()
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    end = time.monotonic() + secs
    n = 0
    ctx = torch.cuda.ExternalStream(stream_handle) if stream_handle is not None else torch.cuda.stream(torch.cuda.default_stream())
    with ctx:
        while time.monotonic() < end:
            a = a @ b
            a = a / a.norm() * 2.0 ** 0.5
            n += 1
    torch.cuda.synchronize()
    RECORDS.append(("matmuls", n))


def main():
    import torch  # noqa: F401
    from holoscan.conditions import CountCondition
    from holoscan.core import Application, Operator, OperatorSpec
    from holoscan.schedulers import EventBasedScheduler

    class SourceOp(Operator):
        def setup(self, spec: OperatorSpec):
            spec.output("out")

        def compute(self, op_input, op_output, context):
            op_output.emit({"branch": "both"}, "out")

    class WorkOp(Operator):
        def __init__(self, fragment, *args, name="work", **kwargs):
            self.work_name = name
            super().__init__(fragment, *args, **kwargs)

        def setup(self, spec: OperatorSpec):
            spec.input("in")
            spec.output("out")

        def compute(self, op_input, op_output, context):
            op_input.receive("in")
            t0 = time.monotonic()
            handle = None
            if VARIANT == "ownstreams":
                handle = context.allocate_cuda_stream(self.work_name + "_s")
            work_gpu(SECS, handle)
            t1 = time.monotonic()
            RECORDS.append((self.work_name, threading.get_ident(), t0, t1, handle))
            op_output.emit({"from": self.work_name}, "out")

    class JoinOp(Operator):
        def setup(self, spec: OperatorSpec):
            spec.input("a")
            spec.input("b")

        def compute(self, op_input, op_output, context):
            ra, rb = op_input.receive("a"), op_input.receive("b")
            RECORDS.append(("join", time.get_ident if False else threading.get_ident(), ra["from"] + rb["from"]))

    app = Application()
    src = SourceOp(app, CountCondition(app, 1), name="src")
    wa = WorkOp(app, name="workA")
    wb = WorkOp(app, name="workB")
    jn = JoinOp(app, CountCondition(app, 2), name="join")
    app.add_flow(src, wa, {("out", "in")})
    app.add_flow(src, wb, {("out", "in")})
    app.add_flow(wa, jn, {("out", "a")})
    app.add_flow(wb, jn, {("out", "b")})

    app.scheduler(EventBasedScheduler(app, worker_thread_number=2, name="ebs"))

    t0 = time.monotonic()
    app.run()
    wall = time.monotonic() - t0

    print(f"\n=== VARIANT={VARIANT} wall={wall:.2f}s (serial ~{2*SECS:.0f}s, concurrent ~{SECS:.0f}s + startup) ===")
    for r in RECORDS:
        if len(r) == 5:
            name, tid, s, e, h = r
            print(f"  {name}: tid={tid % 100000} stream={h} span_dur={e-s:.2f}")
        else:
            print(f"  {r}")


if __name__ == "__main__":
    main()
