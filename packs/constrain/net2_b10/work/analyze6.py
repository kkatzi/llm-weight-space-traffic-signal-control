"""How many free decisions a penalty on through-phase logits would change (first order, recorded states),
and the queue lengths (stop line + upstream, in vehicles) of the through phases the network picks."""
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
can_end = np.where(np.isin(green, [0, 2]), tg >= 15, tg >= 10)
free = can_end & ~overdue.any(1)
CAP_S = 22.4 / 7.5; CAP_U = 256 / 7.5
# vehicles halting per through phase: stop-line lanes 0,1 of the two approaches + upstream lanes
# phase 0: N0,N1,S0,S1 (0,1,6,7) + up N0u,N1u,S0u,S1u (24,25,28,29); phase 2: E0,E1,W0,W1 (3,4,9,10) + up (26,27,30,31)
SL = {0: [0, 1, 6, 7], 2: [3, 4, 9, 10], 1: [2, 8], 3: [5, 11]}
UP = {0: [24, 25, 28, 29], 2: [26, 27, 30, 31]}
def veh(x, p):
    s = x[SL[p]].sum() * CAP_S
    u = x[UP[p]].sum() * CAP_U if p in UP else 0.0
    return s, u
for p in [0, 2]:
    m = np.where(free & (net == p))[0]
    s = np.array([veh(X[i], p)[0] for i in m]); u = np.array([veh(X[i], p)[1] for i in m])
    print(f"pick {p}: stop-line halting veh pct {np.percentile(s, [10, 25, 50, 75, 90]).round(1)}, upstream halting veh pct {np.percentile(u, [10, 25, 50, 75, 90]).round(1)}")
    # max per approach-lane chain (stop-line lane + its upstream lane)
for d in [0.5, 1.0, 1.5, 2.0, 3.0]:
    adj = lg.copy(); adj[:, [0, 2]] -= d
    # only for switching picks (not the current green)
    adj2 = lg.copy()
    for i in range(N):
        for p in (0, 2):
            if p != green[i]:
                adj2[i, p] -= d
    new = adj2.argmax(1)
    ch = free & (new != net)
    print(f"penalty {d}: changed {ch.sum()} of {free.sum()} free decisions;", Counter(zip(net[ch], new[ch])).most_common(6))
