"""Offline analysis: how the network reacts to W2 substitutes on the recorded states."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools
np.set_printoptions(precision=3, suppress=True, linewidth=220)
S = tools.states(); X = S["obs"].astype(float); c = S["controller"]; ep = S["episode"]; t = S["time_s"]
w = tools.load("start_weights.npz")
ok = X[:, 44] > 0.5

def subst(X):
    Y = X.copy(); Y[:, 11] = X[:, 5]; Y[:, 23] = X[:, 17]; return Y

g0 = tools.greedy(w, X); g1 = tools.greedy(w, subst(X))
for ci, nm in enumerate(S["controller_names"]):
    m = (c == ci) & ok
    print(nm, "honoured states", m.sum(), "changed share", (g0[m] != g1[m]).mean(),
          "orig dist", np.bincount(g0[m], minlength=4) / m.sum(), "subst dist", np.bincount(g1[m], minlength=4) / m.sum())
# E vs W side symmetry under fixed-time (detector-independent controller)
m = c == 1
nm = tools.names()
for e, wv in [(3, 9), (4, 10), (5, 11), (15, 21), (16, 22), (17, 23), (26, 30), (27, 31), (34, 38), (35, 39)]:
    print(f"{nm[e]:>6} {X[m, e].mean():.3f}   {nm[wv]:>6} {X[m, wv].mean():.3f}")
