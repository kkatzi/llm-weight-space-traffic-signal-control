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
Gu = {0: [0, 4], 1: [], 2: [2, 6], 3: []}   # upstream lanes exclusively feeding
Gs = {0: [1, 5], 1: [1, 5], 2: [3, 7], 3: [3, 7]}  # shared upstream lane1 (straight+left)
def grp(a, G):
    return np.stack([a[:, G[p]].sum(1) if G[p] else np.zeros(len(a)) for p in range(4)], 1)
Q = grp(q, G); D = grp(d, G); QU = grp(qu, Gu) + grp(qu, Gs); DU = grp(du, Gu) + grp(du, Gs)
g = tools.greedy(w, X)
m = (ctrl == 0) & ok
r = np.arange(len(X))
curQ = Q[r, ph]; curD = D[r, ph]
print("network honoured decisions: current-phase halted veh distribution at decision")
for p in range(4):
    mm = m & (ph == p)
    print(f" phase {p}: n={mm.sum()} curQ mean {curQ[mm].mean():.2f}  frac curQ>=2 {(curQ[mm]>=2).mean():.2f} curD mean {curD[mm].mean():.2f}; stays {(g[mm]==p).sum()}")
# For stays: describe
mm = m & (g == ph)
print("stay cases:", mm.sum())
for i in np.where(mm)[0][:15]:
    print(" ph", ph[i], "tg", round(tg[i]), "Q", Q[i].round(1), "D", D[i].round(1), "QU", QU[i].round(1), "DU", DU[i].round(1))
# logits margin for stays
lg = tools.forward(w, X)
# switching cases where current queue large
mm = m & (g != ph) & (curQ >= 3)
print("switch while current halted >=3:", mm.sum())
for i in np.where(mm)[0][:15]:
    print(" ph", ph[i], "tg", round(tg[i]), "->", g[i], "Q", Q[i].round(1), "D", D[i].round(1), "QU", QU[i].round(1), "DU", DU[i].round(1), "lg", lg[i].round(1))
