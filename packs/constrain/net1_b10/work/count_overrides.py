"""Count how often candidate supervisor rules would change the network's choice on the recorded states.
Descriptive only: no fitting, no search over variants. Usage: python work/count_overrides.py work/supervisor.py"""
import importlib.util
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("sup", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64)
R = S["rule_state"]
L = tools.forward(w, X)
N = len(X)
net = L.argmax(1)
green = R[:, 17].astype(int)
tg = R[:, 16]
legal = R[:, 4:8] > 0.5
rules = {"min_green_through": 15, "min_green_left": 10, "max_green": 45, "max_wait": 90}
out = np.zeros(N, int)
for i in range(N):
    st = {"green": int(green[i]), "t_green": float(tg[i]), "since_served": [float(v) for v in R[i, 0:4]],
          "phase_halting": [float(v) for v in R[i, 12:16]], "legal": [bool(v) for v in legal[i]],
          "overdue": [bool(v) for v in (R[i, 8:12] > 0.5)], "rules": rules}
    if hasattr(mod, "reset") and (i == 0 or S["episode"][i] != S["episode"][i - 1]):
        mod.reset()
    out[i] = mod.decide(X[i], L[i].copy(), st)
ill = ~legal[np.arange(N), out]
print("illegal outputs:", ill.sum())
# effective network choice after shield
def shield(i, a):
    if legal[i, a]:
        return a
    c = np.where(legal[i])[0]
    return c[np.argmax(R[i, 12 + c])]
eff_net = np.array([shield(i, net[i]) for i in range(N)])
print("differs from network argmax:", (out != net).mean().round(4),
      " differs from shielded network:", (out != eff_net).sum(), "decisions")
sw = legal.sum(1) > 1
print("free decisions (>1 legal):", sw.sum(), " changed there:", (out[sw] != net[sw]).sum())
for kind, m in (("to through", np.isin(out, [0, 2])), ("to left", np.isin(out, [1, 3]))):
    print(" changed,", kind, ((out != eff_net) & m).sum())
