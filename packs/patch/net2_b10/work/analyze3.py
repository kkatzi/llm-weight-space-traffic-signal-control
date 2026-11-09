import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=180)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); act = S["act"]; ctrl = S["controller"]; ep = S["episode"]; t = S["time_s"]
cap = 22.4/7.5; capu = 256/7.5
q = X[:, 0:12]*cap; d = X[:, 12:24]*cap; qu = X[:, 24:32]*capu; du = X[:, 32:40]*capu
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5; tg = X[:, 45]*60
G = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
Gu = {0: [0, 1, 4, 5], 1: [1, 5], 2: [2, 3, 6, 7], 3: [3, 7]}
def grp(a, G):
    return np.stack([a[:, G[p]].sum(1) for p in range(4)], 1)
Q = grp(q, G); D = grp(d, G)
g = tools.greedy(w, X)
for c in range(3):
    m = (ctrl == c) & ok
    print("=== ctrl", S["controller_names"][c], "honoured", m.sum())
    print(" stay rate (net greedy == current phase):", (g[m] == ph[m]).mean())
    # net picks phase with max queue (excluding current)?
    Qm = Q[m].copy(); Qm[np.arange(m.sum()), ph[m]] = -1
    print(" net pick == argmax queue among others:", (g[m] == Qm.argmax(1)).mean())
    Dm = D[m].copy(); Dm[np.arange(m.sum()), ph[m]] = -1
    print(" net pick == argmax density among others:", (g[m] == Dm.argmax(1)).mean())
    # queue of chosen phase vs max
    sel = Q[m][np.arange(m.sum()), g[m]]
    print(" mean queue of picked:", sel.mean(), " mean max other queue:", Qm.max(1).mean())
    print(" picks with zero queue & zero density on picked phase:", ((Q[m][np.arange(m.sum()), g[m]] == 0) & (D[m][np.arange(m.sum()), g[m]] < 0.01)).mean())
    # pick distribution given current phase
    for p in range(4):
        mm = m & (ph == p)
        print(f"  cur {p}: n={mm.sum()} picks", np.bincount(g[mm], minlength=4))
