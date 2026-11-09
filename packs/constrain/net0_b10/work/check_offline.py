"""Offline check of work/supervisor.py on the recorded states: legality, and where it differs from the network."""
import sys, importlib.util; sys.path.insert(0, ".")
import numpy as np, tools
path = sys.argv[1] if len(sys.argv) > 1 else "work/supervisor.py"
spec = importlib.util.spec_from_file_location("sup", path); sup = importlib.util.module_from_spec(spec); spec.loader.exec_module(sup)
S = tools.states()
X = S["obs"].astype(float); rs = S["rule_state"]
w = tools.load("start_weights.npz"); L = tools.forward(w, X); net = L.argmax(1)
N = len(X); out = np.zeros(N, int)
for i in range(N):
    st = {"green": int(rs[i, 17]), "t_green": float(rs[i, 16]), "since_served": list(rs[i, 0:4]),
          "phase_halting": list(rs[i, 12:16]), "legal": [bool(v > .5) for v in rs[i, 4:8]],
          "overdue": [bool(v > .5) for v in rs[i, 8:12]], "rules": {}}
    out[i] = sup.decide(X[i], L[i], st)
legal = rs[:, 4:8] > .5
print("all legal:", legal[np.arange(N), out].all())
real = legal.sum(1) > 1
print("differs from network: all states", (out != net).mean().round(3), " at real decisions", (out[real] != net[real]).sum(), "/", real.sum())
d = real & (out != net)
print("  changed from", np.bincount(net[d], minlength=4), "to", np.bincount(out[d], minlength=4))
