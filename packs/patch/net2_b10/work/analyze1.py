import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=180)
w = tools.load("start_weights.npz")
S = tools.states()
print({k: (v.shape, v.dtype) for k, v in S.items()})
print(S["controller_names"] if "controller_names" in S else None)
X = S["obs"]; act = S["act"]; ctrl = S["controller"]; ep = S["episode"]; t = S["time_s"]
names = tools.names()
print("trainability", tools.trainability(w))
for c in np.unique(ctrl):
    m = ctrl == c
    print("controller", c, m.sum(), "episodes", np.unique(ep[m]), "act dist", np.bincount(act[m], minlength=4))
    g = tools.greedy(w, X[m])
    print("  net greedy agrees with recorded act:", (g == act[m]).mean())
# per-feature means per controller
for i, n in enumerate(names):
    print(f"{i:2d} {n:6s}", " ".join(f"{X[ctrl==c, i].mean():.3f}" for c in np.unique(ctrl)))
