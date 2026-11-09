import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
from collections import Counter
np.set_printoptions(precision=2, suppress=True, linewidth=220)
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
# switch opportunities: current green may end
minG = np.array([15, 10, 15, 10])
opp = tg >= minG[green]  # min green elapsed
print("opportunities:", opp.sum())
forced = opp & overdue.any(1)
print("forced decisions (opp & overdue):", forced.sum())
# at forced decisions: overdue phase, network pick, gap
rows = []
for i in np.where(forced)[0]:
    q = np.where(overdue[i])[0]
    qq = q[np.argmax(halt[i, q])]
    a = act[i]
    rows.append((green[i], qq, a, lg[i, a] - lg[i, qq], halt[i, qq], halt[i, a], since[i, qq]))
rows = np.array(rows)
print("forced: net already picks overdue phase share:", (rows[:, 1] == rows[:, 2]).mean())
nz = rows[rows[:, 1] != rows[:, 2]]
print("forced & different: n", len(nz), " gap median", np.median(nz[:, 3]), " mean", nz[:, 3].mean(), "quantiles", np.quantile(nz[:, 3], [.1, .25, .5, .75, .9]))
print("  halting overdue phase: dist", Counter(nz[:, 4].astype(int)))
print("  halting net pick phase: mean", nz[:, 5].mean())
print("  by overdue phase", Counter(nz[:, 1].astype(int)), " by green", Counter(nz[:, 0].astype(int)))
# logit gap in general at opportunities: top - second
srt = np.sort(lg, 1)
gap = srt[:, -1] - srt[:, -2]
print("general opp gap quantiles", np.quantile(gap[opp], [.1, .25, .5, .75, .9]))
# rank of each phase at opportunities
# 'last free chance' states: opp, no overdue, q != act with halt>0 and since[q] >= thr(act)
lfc = []
for i in np.where(opp & ~overdue.any(1))[0]:
    a = act[i]
    if a == green[i]:
        horizon = 5
    else:
        horizon = 15 if a in (1, 3) else 20
    for q in range(4):
        if q == a or q == green[i]:
            continue
        if halt[i, q] > 0 and since[i, q] + horizon >= 90:
            lfc.append((i, green[i], q, a, lg[i, a] - lg[i, q], halt[i, q], halt[i, a], since[i, q]))
lfc = np.array(lfc)
print("last-free-chance states:", len(lfc))
print("  gap quantiles", np.quantile(lfc[:, 4], [.1, .25, .5, .75, .9]))
print("  q dist", Counter(lfc[:, 2].astype(int)), " since dist", Counter(lfc[:, 7].astype(int)))
print("  halt q dist", Counter(lfc[:, 5].astype(int)))
