import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
from collections import Counter
np.set_printoptions(precision=3, suppress=True, linewidth=220)
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
# overdue threshold
for p in range(4):
    m = green != p
    od = overdue[:, p] & m
    nod = (~overdue[:, p]) & m & (halt[:, p] > 0)
    print(p, "overdue since min/max", since[od, p].min() if od.any() else None, since[od, p].max() if od.any() else None,
          " non-overdue w/ halt since max", since[nod, p].max())
    print("   overdue with halt==0:", (od & (halt[:, p] == 0)).sum(), " since>=90 & halt>0 & not overdue:", ((since[:, p] >= 90) & (halt[:, p] > 0) & ~overdue[:, p] & m).sum())
    print("   since values with halt>0, not overdue, >=80:", Counter(since[nod & (since[:, p] >= 80), p].astype(int)))
    print("   since values overdue:", Counter(since[od, p].astype(int)))
# halting counts vs queue obs
qcols = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
cap_stop = 22.4 / 7.5
for p in range(4):
    approx = X[:, qcols[p]].sum(1) * cap_stop
    print("phase", p, "halt vs sum(q)*cap corr", np.corrcoef(approx, halt[:, p])[0, 1], " max abs diff", np.abs(approx - halt[:, p]).max())
