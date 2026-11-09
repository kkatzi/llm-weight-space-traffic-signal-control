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
# consistency of choice at through tg=12 vs tg=17 (consecutive rows, same episode, same green)
same = 0; tot = 0; pairs = Counter()
for i in range(N - 1):
    if ep[i] == ep[i+1] and green[i] in (0, 2) and int(tg[i]) == 12 and int(tg[i+1]) == 17 and green[i+1] == green[i]:
        if overdue[i+1].any():
            continue
        tot += 1; same += act[i] == act[i+1]; pairs[(act[i], act[i+1])] += 1
print("through 12->17 same choice:", same, "/", tot)
print(sorted(pairs.items()))
# also 7 -> 12 for left phases
same = 0; tot = 0
for i in range(N - 1):
    if ep[i] == ep[i+1] and green[i] in (1, 3) and int(tg[i]) == 7 and int(tg[i+1]) == 12 and green[i+1] == green[i]:
        if overdue[i+1].any():
            continue
        tot += 1; same += act[i] == act[i+1]
print("left 7->12 same choice:", same, "/", tot)
same = 0; tot = 0
for i in range(N - 1):
    if ep[i] == ep[i+1] and green[i] in (0, 2) and int(tg[i]) == 7 and int(tg[i+1]) == 12 and green[i+1] == green[i]:
        tot += 1; same += act[i] == act[i+1]
print("through 7->12 same choice:", same, "/", tot)
# Network's choice vs max-halting among others at opportunities
opp = (tg >= np.array([15, 10, 15, 10])[green]) & ~overdue.any(1)
others_h = halt.copy(); others_h[np.arange(N), green] = -1
mh = others_h.argmax(1)
print("opp(no overdue)", opp.sum(), " net choice == most-halting other:", (act[opp] == mh[opp]).mean())
# stay at opp?
print("stay share at opp:", (act[opp] == green[opp]).mean())
# what is total halting distribution per phase at opp
for p in range(4):
    print("phase", p, "halting mean", halt[:, p].mean().round(2), " when chosen at opp:", halt[opp & (act == p), p].mean().round(2), " n chosen", (opp & (act == p)).sum())
