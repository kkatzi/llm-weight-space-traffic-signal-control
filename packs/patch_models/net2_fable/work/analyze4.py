"""Does the network leave the current phase while its upstream lanes still hold a halted queue?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import demand_matrix, UPSHARE, CAP_UP

w = tools.load("start_weights.npz")
S = tools.states()
X, act, ep, ctl, t = S["obs"], S["act"], S["episode"], S["controller"], S["time_s"]
g = tools.greedy(w, X)
ok = X[:, 44] > 0.5
phase = X[:, 40:44].argmax(1)
U = demand_matrix()
u = X @ U.T
# upstream halting cars attributable to each phase
Q = np.zeros((4, 46))
for l, shares in UPSHARE.items():
    for p, sh in shares:
        Q[p, 24 + l] += sh * CAP_UP
uq = X @ Q.T
idx = np.arange(len(X))
for c in range(3):
    m = (ctl == c) & ok & ((g if c == 0 else act) != phase)
    a = g if c == 0 else act
    cur_uq = uq[m][np.arange(m.sum()), phase[m]]
    cur_u = u[m][np.arange(m.sum()), phase[m]]
    print(f"ctl {c}: switches {m.sum()}, current-phase upstream halting cars: mean {cur_uq.mean():.2f}, "
          f"share>=2: {(cur_uq >= 2).mean():.3f}, share>=4: {(cur_uq >= 4).mean():.3f};  current u mean {cur_u.mean():.2f}")
    big = m.copy(); big[m] = cur_uq >= 3
    if big.any():
        best = u[big].argmax(1)
        print("   when cur upstream halting >=3: n", big.sum(), " current phase is argmax u:", (best == phase[big]).mean(),
              " chosen phase u mean:", u[big][np.arange(big.sum()), a[big]].mean(), " cur u mean:", u[big][np.arange(big.sum()), phase[big]].mean(),
              " by phase:", np.bincount(phase[big], minlength=4))

# Time profile of upstream halting (all phases) for start-net episodes 3,4 (busiest)
m = (ctl == 0) & np.isin(ep, [3, 4])
tb = (t[m] // 300).astype(int)
tot_uq = uq[m].sum(1)
print("\nmean total upstream halting cars per 5-min bin (busy start-net episodes):")
print([round(tot_uq[tb == b].mean(), 2) for b in range(tb.max() + 1)])
tot_sl = (X[m, 12:24] * (22.4 / 7.5)).sum(1)
print("mean total stop-line cars per 5-min bin:")
print([round(tot_sl[tb == b].mean(), 2) for b in range(tb.max() + 1)])
# per-phase stop-line saturation: how often all stop-line lanes of EW through are full (d=1) at honoured time
full = (X[:, 12 + np.array([3, 4, 9, 10])] >= 0.99).sum(1)
print("EW-through stop-line lanes full count at honoured states (start-net):", np.bincount(full[(ctl == 0) & ok], minlength=5))
