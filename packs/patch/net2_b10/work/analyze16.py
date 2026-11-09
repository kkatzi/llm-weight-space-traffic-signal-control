import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
from collections import Counter
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]; ep = S["episode"]
cap, capu = 22.4/7.5, 256/7.5
q = X[:, 0:12]*cap; d = X[:, 12:24]*cap; du = X[:, 32:40]*capu
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
G = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
Q = np.stack([q[:, G[p]].sum(1) for p in range(4)], 1); D = np.stack([d[:, G[p]].sum(1) for p in range(4)], 1)
g = tools.greedy(w, X); r = np.arange(len(X))
m = ok & (ctrl == 0)
D2 = D.copy(); D2[r, ph] = -1
best_other = D2.max(1)
picked = D[r, g]
print("net honoured: picked phase vehicles (stop-line D) vs max other")
tab = Counter(zip(np.round(picked[m]).astype(int).clip(0, 6), np.round(best_other[m]).astype(int).clip(0, 9)))
for pk in range(0, 4):
    print(f" picked D={pk}: ", {bo: tab[(pk, bo)] for bo in range(10) if tab[(pk, bo)]})
mm = m & (picked <= 1.01) & (best_other >= 4)
print("picked <=1 veh while another phase has >=4:", mm.sum(), "/", m.sum())
for i in np.where(mm)[0][:12]:
    print("  ph", ph[i], "->", g[i], "D", D[i].round(0), "Q", Q[i].round(0), "du", du[i].round(0))
