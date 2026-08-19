"""Live probe: does holoscan-cu13 4.2 schedule independent DAG branches concurrently?

Three variants (run as subprocesses):
  default    - GreedyScheduler (the 4.2 default)
  multithread- MultiThreadScheduler(worker_thread_number=2)
  eventbased - EventBasedScheduler(worker_thread_number=2)

DAG: source -> workA -> join ; source -> workB -> join
Each WorkOp records (name, thread id, start, end) and either time.sleep(SECS)
(mode=cpu) or runs ~SECS of GPU matmul work (mode=gpu, releases the GIL).
"""
import sys
import threading
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
SCHED = sys.argv[2] if len(sys.argv) > 2 else "default"
SECS = 2.0

RECORDS = []


def work_cpu(secs):
    time.sleep(secs)


def work_gpu(secs):
    import torch

    torch.cuda.init()
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    end = time.monotonic() + secs
    n = 0
    while time.monotonic() < end:
        a = a @ b
        a = a / a.norm() * 2.0 ** 0.5
        n += 1
    RECORDS.append(("matmuls", n))


def main():
    import torch  # noqa: F401  (import early, same as the app)
    from holoscan.conditions import CountCondition
    from holoscan.core import Application, Operator, OperatorSpec
    from holoscan.schedulers import EventBasedScheduler, MultiThreadScheduler

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
            if MODE == "cpu":
                work_cpu(SECS)
            else:
                work_gpu(SECS)
            t1 = time.monotonic()
            RECORDS.append((self.work_name, threading.get_ident(), t0, t1))
            op_output.emit({"from": self.work_name}, "out")

    class JoinOp(Operator):
        def setup(self, spec: OperatorSpec):
            spec.input("a")
            spec.input("b")

        def compute(self, op_input, op_output, context):
            ra, rb = op_input.receive("a"), op_input.receive("b")
            RECORDS.append(("join", threading.get_ident(), *sorted(ra["from"] + rb["from"])))

    app = Application()
    src = SourceOp(app, CountCondition(app, 1), name="src")
    wa = WorkOp(app, name="workA")
    wb = WorkOp(app, name="workB")
    jn = JoinOp(app, CountCondition(app, 2), name="join")
    app.add_flow(src, wa, {("out", "in")})
    app.add_flow(src, wb, {("out", "in")})
    app.add_flow(wa, jn, {("out", "a")})
    app.add_flow(wb, jn, {("out", "b")})

    if SCHED == "multithread":
        app.scheduler(MultiThreadScheduler(app, worker_thread_number=2, name="mts"))
    elif SCHED == "eventbased":
        app.scheduler(EventBasedScheduler(app, worker_thread_number=2, name="ebs"))

    t0 = time.monotonic()
    app.run()
    wall = time.monotonic() - t0

    print(f"\n=== SCHED={SCHED} MODE={MODE} wall={wall:.2f}s (expected ~{2*SECS:.0f}s serial, ~{SECS:.0f}s concurrent) ===")
    for r in RECORDS:
        if len(r) == 4 and isinstance(r[1], int) and r[1] > 10**10:  # thread id present
            name, tid, s, e = r
            print(f"  {name}: tid={tid} span=[{s:.3f}, {e:.3f}] dur={e-s:.2f}")
        else:
            print(f"  {r}")


if __name__ == "__main__":
    main()
