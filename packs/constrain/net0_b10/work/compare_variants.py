"""Check that work/supervisor.py makes exactly the same decisions as the evaluated variant sup_v3_pen4_k4.py on the
recorded states (so the final file is the same policy as evaluation 4), plus a few edge cases."""
import sys, importlib.util; sys.path.insert(0, ".")
import numpy as np, tools
def load(p):
    s = importlib.util.spec_from_file_location(p, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.decide
a, b = load("work/supervisor.py"), load("work/sup_v3_pen4_k4.py")
S = tools.states(); X = S["obs"].astype(float); rs = S["rule_state"]
L = tools.forward(tools.load("start_weights.npz"), X)
diff = 0
for i in range(len(X)):
    st = {"green": int(rs[i, 17]), "t_green": float(rs[i, 16]), "since_served": list(rs[i, 0:4]),
          "phase_halting": list(rs[i, 12:16]), "legal": [bool(v > .5) for v in rs[i, 4:8]],
          "overdue": [bool(v > .5) for v in rs[i, 8:12]], "rules": {}}
    diff += a(X[i], L[i], st) != b(X[i], L[i], st)
print("decisions differing between final and v3:", diff)
# edge cases: max-green style state (current illegal, others legal), two overdue phases, numpy-typed state
st = {"green": 2, "t_green": 47, "legal": np.array([1, 1, 0, 1]), "overdue": np.array([0, 0, 0, 0]),
      "phase_halting": np.array([3, 1, 0, 2]), "since_served": [50, 60, 0, 40], "rules": {}}
print("max-green case ->", a(X[0], np.array([0.1, 0.5, 3.0, 0.2]), st), "(expect 1: best legal by logits)")
st2 = dict(st, legal=[True, True, False, False], overdue=[True, True, False, False])
print("two overdue ->", a(X[0], np.array([0.1, 0.5, 3.0, 0.2]), st2), "(expect 1)")
