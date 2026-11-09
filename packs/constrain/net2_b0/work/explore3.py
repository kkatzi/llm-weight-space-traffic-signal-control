import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
from collections import Counter
np.set_printoptions(precision=3, suppress=True, linewidth=200)
S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
lg = tools.forward(w, X)
act = S["act"].astype(int)
rs = S["rule_state"]
since = rs[:, 0:4]; legal = rs[:, 4:8] > 0.5; overdue = rs[:, 8:12] > 0.5; halt = rs[:, 12:16]
tg = rs[:, 16]; green = rs[:, 17].astype(int)
ep = S["episode"]; t = S["time_s"]
N = len(X)
# green durations: find runs per episode
durs = Counter(); trans = Counter()
for e in np.unique(ep):
    idx = np.where(ep == e)[0]
    g = green[idx]; tt = tg[idx]
    for k in range(1, len(idx)):
        if g[k] != g[k-1]:
            durs[(g[k-1], int(tt[k-1]))] += 1  # last decision tg at which switch requested
            trans[(g[k-1], g[k])] += 1
print("green ended at decision tg (phase, tg):")
for p in range(4):
    print(p, sorted([(k[1], v) for k, v in durs.items() if k[0] == p]))
print("transitions:", sorted(trans.items()))
# episode lengths
for e in np.unique(ep):
    idx = np.where(ep == e)[0]
    print("ep", e, "n", len(idx), "t range", t[idx].min(), t[idx].max())
# at decisions where switch is legal (all legal), what does the net pick
al = legal.all(1)
print("all-legal states:", al.sum(), " net picks current (stay):", (act[al] == green[al]).mean())
print("all-legal by green,tg -> stay share")
for p in range(4):
    for tv in sorted(set(tg[al & (green == p)].astype(int))):
        m = al & (green == p) & (tg.astype(int) == tv)
        print(p, tv, m.sum(), "stay", round((act[m] == p).mean(), 3), "picks", Counter(act[m]))
