"""(1) How much of the forced through extension (12 s -> 17 s) is used? (2) How often would near-tie rules fire?"""
import os, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools

S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
R = S["rule_state"]
ep = S["episode"]; ts = S["time_s"]
lg = tools.forward(w, X)
net = lg.argmax(1)
since = R[:, 0:4]; legal = R[:, 4:8] > 0.5; overdue = R[:, 8:12] > 0.5
ph = R[:, 12:16]; tg = R[:, 16]; green = R[:, 17].astype(int)
N = len(X)
free = legal.all(1)

# stop-line lanes per phase (indices into 0..11) and upstream lanes (0..7) feeding them
SL = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
UP = {0: [0, 1, 4, 5], 2: [2, 3, 6, 7]}
cap_sl = 22.4 / 7.5
cap_up = 256 / 7.5

for t in (12, 17):
    m = (green % 2 == 0) & (tg == t)
    sl = np.array([X[i, 12 + np.array(SL[green[i]])].sum() * cap_sl for i in np.where(m)[0]])
    up = np.array([X[i, 32 + np.array(UP[green[i]])].sum() * cap_up for i in np.where(m)[0]])
    upq = np.array([X[i, 24 + np.array(UP[green[i]])].sum() * cap_up for i in np.where(m)[0]])
    print(f"through green at t={t}: n={m.sum()}, stop-line veh mean {sl.mean():.2f}, upstream veh mean {up.mean():.2f}, "
          f"upstream halting mean {upq.mean():.2f}, halting own phase mean {ph[m, :][np.arange(m.sum()), green[m]].mean():.2f}")
# same for left phases at t=7 and 12
for t in (7, 12):
    m = (green % 2 == 1) & (tg == t)
    sl = np.array([X[i, 12 + np.array(SL[green[i]])].sum() * cap_sl for i in np.where(m)[0]])
    print(f"left green at t={t}: n={m.sum()}, stop-line veh mean {sl.mean():.2f}, halting own {ph[m, :][np.arange(m.sum()), green[m]].mean():.2f}")

# near ties at free decisions where the network switches
fs = np.where(free & (net != green))[0]
print("free switch decisions:", len(fs))
for delta in (0.5, 1.0, 1.5):
    n_lt = 0; n_od = 0
    for i in fs:
        c = net[i]
        alts = [p for p in range(4) if p != c and p != green[i] and lg[i, c] - lg[i, p] <= delta]
        if c % 2 == 0 and any(p % 2 == 1 for p in alts):
            n_lt += 1
        hor = 20 if c % 2 == 0 else 15
        if any(since[i, p] + hor >= 90 and ph[i, p] > 0 for p in alts):
            n_od += 1
    print(f"delta {delta}: through-top with left within delta: {n_lt}; soon-overdue alt within delta: {n_od}")

# horizon check: does the network's next free decision come exactly 20 s (through) / 15 s (left) later?
# since[p] at a forced-overdue decision
print("since of overdue phase when it first becomes overdue:", Counter([int(since[i][overdue[i]].min()) for i in range(N) if overdue[i].any() and (i == 0 or not overdue[i-1].any())]))
