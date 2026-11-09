"""Statistics of the recorded run of the network under the rules (shield only)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools

S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
act = S["act"].astype(int)
rs = S["rule_state"]
since = rs[:, 0:4]; legal = rs[:, 4:8] > 0.5; overdue = rs[:, 8:12] > 0.5; ph = rs[:, 12:16]
tg = rs[:, 16]; green = rs[:, 17].astype(int)
ep = S["episode"]; t = S["time_s"]
lg = tools.forward(w, X)
net = lg.argmax(1)
print("act == net argmax:", (act == net).mean())
print("t_green values:", np.unique(tg, return_counts=True))
ill = ~legal[np.arange(len(act)), act]
print("illegal requests:", ill.sum(), "of", len(act))
# patterns of legal mask
from collections import Counter
pat = Counter(tuple(l.astype(int)) for l in legal)
print("legal patterns:", pat.most_common(12))
# reasons: classify
nlegal = legal.sum(1)
cur_legal = legal[np.arange(len(act)), green]
print("states where current green is not legal (must switch):", (~cur_legal).sum())
print("states with any overdue:", overdue.any(1).sum())
# illegal requests by t_green and whether request == green
for name, m in [("req stay (=green)", act == green), ("req switch", act != green)]:
    mm = ill & m
    print(name, "illegal:", mm.sum(), "t_green:", Counter(tg[mm].astype(int)).most_common(10))
# min green blocks: through phase at t_green 12 wanting to switch
thr = np.isin(green, [0, 2])
m = thr & (tg == 12)
print("through at tg=12:", m.sum(), "net wants switch:", (m & (net != green)).sum())
# forced switches by max green / overdue
mg = (~cur_legal) & (tg > 45)
print("forced switch with tg>45:", mg.sum())
od = (~cur_legal) & (tg <= 45)
print("forced switch with tg<=45 (overdue):", od.sum())
# compare shield choice (legal, most halting) vs network's best legal phase on illegal requests
idx = np.where(ill)[0]
agree = 0; diffs = []
for i in idx:
    L = np.where(legal[i])[0]
    sh = L[np.argmax(ph[i, L])]
    nb = L[np.argmax(lg[i, L])]
    agree += sh == nb
    diffs.append((green[i], int(tg[i]), act[i], sh, nb, tuple(L), tuple(ph[i].astype(int)), tuple(np.round(lg[i], 2))))
print("shield choice == net best legal:", agree, "/", len(idx))
for d in diffs[:40]:
    print(d)
