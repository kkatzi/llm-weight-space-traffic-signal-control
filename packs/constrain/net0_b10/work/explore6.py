import sys; sys.path.insert(0, ".")
import numpy as np, tools
S = tools.states()
X = S["obs"].astype(float); act = S["act"].astype(int); rs = S["rule_state"]
w = tools.load("start_weights.npz")
L = tools.forward(w, X)
ss = rs[:, 0:4]; legal = rs[:, 4:8] > 0.5; ph = rs[:, 12:16]; tg = rs[:, 16]; g = rs[:, 17].astype(int)
N = len(X)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
ok = X[:, 44] > 0.5
dec = ok & legal.all(1)
LANES = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
UP = {0: [24, 25, 28, 29], 1: [25, 29], 2: [26, 27, 30, 31], 3: [27, 31]}
def scaled(x, p, f):
    x = x.copy()
    for l in LANES[p]:
        x[l] *= f; x[12 + l] *= f
    for u in UP[p]:
        x[u] *= f; x[8 + u] *= f
    return x
for P in [(0, 2), (1, 3)]:
    idx = np.where(dec & np.isin(act, P) & (act != g))[0]
    d = []
    for i in idx:
        p = act[i]
        x2 = scaled(X[i], p, 0.75)
        l2 = tools.forward(w, x2)[0]
        d.append((L[i, p] - l2[p], (L[i, p] - np.delete(L[i], p).max()) - (l2[p] - np.delete(l2, p).max())))
    d = np.array(d)
    print("phases", P, "n", len(idx), "drop own logit at 0.75x queue: quantiles", np.quantile(d[:, 0], [.1, .25, .5, .75, .9]),
          " drop of gap:", np.quantile(d[:, 1], [.1, .25, .5, .75, .9]))
    # how many decisions would flip if the queue were 0.75x
    flips = 0
    for i in idx:
        l2 = tools.forward(w, scaled(X[i], act[i], 0.75))[0]
        flips += l2.argmax() != act[i]
    print("  flips at 0.75x:", flips, "/", len(idx))
