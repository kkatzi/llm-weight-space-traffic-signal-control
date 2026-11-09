"""First look: how the start network reacts to W2 information, on recorded states."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools

np.set_printoptions(precision=3, suppress=True, linewidth=160)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64)
ctrl = S["controller"]
nm = tools.names()
ok = X[:, 44] > 0.5

a0 = tools.greedy(w, X)
print("action dist per controller-data (greedy of start net), ok states")
for c in range(3):
    m = (ctrl == c) & ok
    print(c, np.bincount(a0[m], minlength=4) / m.sum())

# recorded action dist of start network
m = ctrl == 0
print("recorded act start net", np.bincount(S["act"][m], minlength=4) / m.sum())
print("greedy==recorded on start-net states", (a0[m] == S["act"][m]).mean())

# imputation: copy E2 into W2
Xi = X.copy()
Xi[:, 11] = X[:, 5]
Xi[:, 23] = X[:, 17]
a1 = tools.greedy(w, Xi)
for c in range(3):
    mm = (ctrl == c) & ok
    print("ctrl", c, "changed decisions (ok) with W2:=E2", (a0[mm] != a1[mm]).mean(),
          "dist", np.bincount(a1[mm], minlength=4) / mm.sum())

# sweep q_W2, d_W2 jointly from 0..1 on start-net ok states: share choosing phase 3
for v in [0, 0.33, 0.67, 1.0]:
    Xv = X.copy()
    Xv[:, 11] = v
    Xv[:, 23] = v
    av = tools.greedy(w, Xv)
    mm = ok
    print("W2=", v, "dist", np.bincount(av[mm], minlength=4) / mm.sum())

# logit sensitivities: d logits / d q_W2 and d_W2 averaged
def grad_in(w, X, j, eps=1e-4):
    Xp = X.copy(); Xp[:, j] += eps
    return (tools.forward(w, Xp) - tools.forward(w, X)) / eps
for j in [5, 17, 11, 23, 2, 8, 31, 39, 27, 35]:
    g = grad_in(w, X, j)
    print(nm[j], "mean dlogit", g[ok].mean(0))
