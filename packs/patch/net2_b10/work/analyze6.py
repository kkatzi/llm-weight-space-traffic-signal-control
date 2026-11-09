import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
from collections import Counter
np.set_printoptions(precision=2, suppress=True, linewidth=180)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]; ep = S["episode"]; t = S["time_s"]
cap = 22.4/7.5; capu = 256/7.5
q = X[:, 0:12]*cap; d = X[:, 12:24]*cap; qu = X[:, 24:32]*capu; du = X[:, 32:40]*capu
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5; tg = X[:, 45]*60
G = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
def grp(a, G):
    return np.stack([a[:, G[p]].sum(1) for p in range(4)], 1)
Q = np.round(grp(q, G)); D = np.round(grp(d, G))
lg = tools.forward(w, X); g = lg.argmax(1)
r = np.arange(len(X))
m = ok
# picked phase has no vehicles at all on its stop-line lanes
empty_pick = (D[r, g] == 0) & (g != ph)
others = Q.copy(); others[r, ph] = -1
print("honoured: picks phase with D=0 while some other non-current phase has Q>0:", (m & empty_pick & (others.max(1) > 0)).sum(), "of", m.sum())
for c in range(3):
    mm = m & (ctrl == c)
    print(" ctrl", c, (mm & empty_pick & (others.max(1) > 0)).sum(), "/", mm.sum())
idx = np.where(m & empty_pick & (others.max(1) > 0) & (ctrl == 0))[0]
for i in idx[:20]:
    print(" ph", ph[i], "->", g[i], "Q", Q[i], "D", D[i], "qu", np.round(qu[i]), "du", np.round(du[i]), "lg", lg[i].round(1))
# Distribution of Q by phase picked vs alternative
print("\nnetwork-episode honoured decisions; table of (Q of pick) - (max Q of other non-current):")
mm = m & (ctrl == 0) & (g != ph)
diff = Q[r, g] - np.where(np.arange(4)[None, :] == g[:, None], -99, others).max(1)
print(Counter(diff[mm].astype(int)))
# where pick is much worse than alternative
idx = np.where(mm & (diff <= -3))[0]
print("cases pick Q <= alt-3:", len(idx))
for i in idx[:25]:
    print(" ph", ph[i], "->", g[i], "Q", Q[i], "D", D[i], "lanesQ", np.round(q[i]).astype(int), "du", np.round(du[i]).astype(int), "lg", lg[i].round(1))
