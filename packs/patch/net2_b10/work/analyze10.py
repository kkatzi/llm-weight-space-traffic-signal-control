import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]
ok = X[:, 44] > 0.5
# mirror N<->S and E<->W
perm = list(range(46))
def swap(a, b):
    perm[a], perm[b] = perm[b], perm[a]
for base in [0, 12]:
    for l in range(3):
        swap(base + 0*3 + l, base + 2*3 + l)  # N<->S
        swap(base + 1*3 + l, base + 3*3 + l)  # E<->W
for base in [24, 32]:
    for l in range(2):
        swap(base + 0*2 + l, base + 2*2 + l)
        swap(base + 1*2 + l, base + 3*2 + l)
Xm = X[:, perm]
names = tools.names()
print([names[i] for i in perm[:24]])
g = tools.greedy(w, X); gm = tools.greedy(w, Xm)
print("mirror agreement honoured:", (g[ok] == gm[ok]).mean(), "all:", (g == gm).mean())
lg = tools.forward(w, X); lgm = tools.forward(w, Xm)
print("mean abs logit diff:", np.abs(lg - lgm)[ok].mean(0))
# input sensitivity: finite diff per input on honoured states, logits
eps = 1/3
for i in range(40):
    Xp = X[ok].copy(); Xp[:, i] = np.clip(Xp[:, i] + (eps if i < 24 else 1/34), 0, 1)
    d = (tools.forward(w, Xp) - lg[ok]).mean(0)
    print(f"{names[i]:6s}", d.round(2))
