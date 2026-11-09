"""Offline check of work/supervisor.py on the recorded states (first order: the recorded trajectory does not react).
Reports: requests that would be illegal (left to the shield), changes vs the network at decisions that matter."""
import importlib.util
import os, sys
import numpy as np
from collections import Counter
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "work", "supervisor.py")
spec = importlib.util.spec_from_file_location("sup", path)
sup = importlib.util.module_from_spec(spec); spec.loader.exec_module(sup)

S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
rs = S["rule_state"]
lg = tools.forward(w, X)
net = lg.argmax(1)
N = len(net)
out = np.empty(N, int)
legal = rs[:, 4:8] > 0.5
for i in range(N):
    st = {"green": int(rs[i, 17]), "t_green": float(rs[i, 16]), "since_served": rs[i, 0:4].tolist(),
          "phase_halting": rs[i, 12:16].tolist(), "legal": legal[i].tolist(), "overdue": (rs[i, 8:12] > 0.5).tolist(),
          "rules": {}}
    out[i] = sup.decide(X[i], lg[i], st)
nlegal = legal.sum(1)
multi = nlegal > 1
print("all states:", N, " differs from network:", (out != net).sum(), f"({(out != net).mean():.3f})")
print("decisions with >1 legal phase:", multi.sum(), " differs there:", (out[multi] != net[multi]).sum())
print("illegal outputs at multi-legal decisions:", (~legal[np.arange(N), out] & multi).sum())
print("changes (net -> out):", Counter(zip(net[out != net], out[out != net])).most_common())
