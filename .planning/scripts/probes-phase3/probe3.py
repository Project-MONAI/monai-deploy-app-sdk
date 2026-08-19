import sys, threading, time
SECS = 1.5
RECORDS = []
def main():
    from holoscan.conditions import CountCondition
    from holoscan.core import Application, Operator, OperatorSpec
    from holoscan.schedulers import EventBasedScheduler
    class Source(Operator):
        def setup(self, spec): spec.output("out")
        def compute(self, i, o, c): o.emit({}, "out")
    class Work(Operator):
        def __init__(self, f, *a, name="w", **k):
            self.n = name
            super().__init__(f, *a, **k)
        def setup(self, spec):
            spec.input("in"); spec.output("out")
        def compute(self, i, o, c):
            i.receive("in"); t0 = time.monotonic()
            time.sleep(SECS)
            RECORDS.append((self.n, threading.get_ident(), t0, time.monotonic()))
            o.emit({}, "out")
    class Join(Operator):
        def setup(self, spec):
            for p in ("a","b","c"): spec.input(p)
        def compute(self, i, o, c):
            for p in ("a","b","c"): i.receive(p)
            RECORDS.append(("join", threading.get_ident(), time.monotonic(), None))
    app = Application()
    s = Source(app, CountCondition(app, 1), name="s")
    w1, w2, w3 = Work(app, name="w1"), Work(app, name="w2"), Work(app, name="w3")
    j = Join(app, CountCondition(app, 3), name="j")
    app.add_flow(s, w1, {("out","in")}); app.add_flow(s, w2, {("out","in")}); app.add_flow(s, w3, {("out","in")})
    app.add_flow(w1, j, {("out","a")}); app.add_flow(w2, j, {("out","b")}); app.add_flow(w3, j, {("out","c")})
    app.scheduler(EventBasedScheduler(app, worker_thread_number=3, name="ebs"))
    t0 = time.monotonic(); app.run(); wall = time.monotonic() - t0
    print(f"=== 3-branch wall={wall:.2f}s (serial {3*SECS:.1f}s, concurrent ~{SECS:.1f}s) joins={sum(1 for r in RECORDS if r[0]=='join')}")
    for r in RECORDS:
        if r[0] != "join": print(f"  {r[0]} tid={r[1]} dur={r[3]-r[2]:.2f}")
main()
