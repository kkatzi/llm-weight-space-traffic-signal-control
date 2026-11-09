"""At free decisions: which phases will be overdue by the time the network's pick can end (look-ahead),
and how sensitive the logits are to queue inputs (finite differences on the recorded states)."""
import os, sys
import numpy as np
from collections import Counter
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools

S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
rs = S["rule_state"]
since = rs[:, 0:4]; legal = rs[:, 4:8] > 0.5; overdue = rs[:, 8:12] > 0.5; ph = rs[:, 12:16]
tg = rs[:, 16].astype(int); green = rs[:, 17].astype(int)
lg = tools.forward(w, X)
net = lg.argmax(1)
N = len(net)
DEND = {0: 20, 1: 15, 2: 20, 3: 15}     # from the switch decision to the first decision the new green may end
can_end = np.where(np.isin(green, [0, 2]), tg >= 15, tg >= 10)
free = can_end & ~overdue.any(1)
cnt = Counter(); nM = Counter()
for i in np.where(free)[0]:
    p = net[i]
    d = 5 if p == green[i] else DEND[p]
    M = [q for q in range(4) if q != p and q != green[i] and since[i, q] + d >= 90 and ph[i, q] > 0]
    M0 = [q for q in range(4) if q != p and q != green[i] and since[i, q] + d >= 90]
    nM[len(M)] += 1
    if M:
        cnt[(int(green[i]), int(p), tuple(M))] += 1
print("free decisions:", free.sum(), "size of must-serve set:", sorted(nM.items()))
print("(green, pick, must-serve):", cnt.most_common(20))

# logit sensitivity: add one halting vehicle (queue+density) on a stop-line lane
# stop-line lane capacity = 22.4/7.5 = 2.99 vehicles; one vehicle = 1/2.99 of the input
one = 7.5 / 22.4
lanes = {"N0": 0, "N1": 1, "N2": 2, "E0": 3, "E1": 4, "E2": 5, "S0": 6, "S1": 7, "S2": 8, "W0": 9, "W1": 10, "W2": 11}
base = lg[free]
for nm, j in lanes.items():
    Xp = X[free].copy()
    Xp[:, j] = np.minimum(1, Xp[:, j] + one)
    Xp[:, 12 + j] = np.minimum(1, Xp[:, 12 + j] + one)
    d = tools.forward(w, Xp) - base
    print(f"+1 veh on {nm}: mean d logits {d.mean(0).round(2)}")
# t_green sensitivity: logit change from tg 12->17 feature (0.2 -> 0.283)
Xp = X[free].copy(); Xp[:, 45] = Xp[:, 45] + 5 / 60
print("t_green +5s: mean d logits", (tools.forward(w, Xp) - base).mean(0).round(2))
