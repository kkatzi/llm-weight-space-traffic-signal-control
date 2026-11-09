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
ep = S["episode"]
N = len(X)
D = np.array([20, 15, 20, 15])
free = legal.all(1) & ~overdue.any(1)
cases = []
for i in np.where(free)[0]:
    a = act[i]
    hor = 5 if a == green[i] else D[a]
    for q in range(4):
        if q in (a, green[i]) or halt[i, q] <= 0:
            continue
        if since[i, q] + hor >= 90:
            smith = halt[i, q] * hor / 15.0 if a == green[i] else halt[i, q] * D[a]
            cases.append((i, q, a, halt[i, q], halt[i, a], lg[i, a] - lg[i, q], halt[i, q] * D[a] >= halt[i, a] * D[q]))
cases = np.array(cases, dtype=float)
print("at-risk (would be forced at next opportunity) cases:", len(cases))
sm = cases[:, 6] > 0
print(" Smith favours q now:", sm.sum(), " gaps there:", np.round(np.sort(cases[sm, 5]), 2))
print(" W_q, W_r there:", cases[sm][:, 3:5].T)
# what happens next in those cases: is q forced at next opportunity in the recorded run?
# check next opportunity row
for row in cases[sm][:10]:
    i = int(row[0]); q = int(row[1])
    j = i + 1
    while j < N and ep[j] == ep[i] and not (tg[j] >= np.array([15, 10, 15, 10])[green[j]]):
        j += 1
    if j < N and ep[j] == ep[i]:
        print("   i", i, "q", q, "r", int(row[2]), "next opp row", j, "overdue", overdue[j].astype(int), "net pick", act[j], "halt", halt[j])
