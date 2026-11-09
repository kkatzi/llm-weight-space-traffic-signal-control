"""Spill-back states (upstream halting >= 4 vehicles on a lane): which lanes, what is green, what the network picks."""
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
ep = S["episode"]; t = S["time_s"]
lg = tools.forward(w, X)
net = lg.argmax(1)
N = len(net)
names = tools.names()
CAP_U = 256 / 7.5; CAP_S = 22.4 / 7.5
up = X[:, 24:32] * CAP_U
can_end = np.where(np.isin(green, [0, 2]), tg >= 15, tg >= 10)
free = can_end & ~overdue.any(1)
sp = up.max(1) >= 4
print("spill states:", sp.sum(), " by lane:", Counter(np.array(names[24:32])[up[sp].argmax(1)]).most_common())
m = sp & free
print("free decisions in spill:", m.sum(), " net picks:", Counter(net[m]).most_common(), " green:", Counter(green[m]).most_common())
# pocket occupancy (density) for left lanes when E1u/W1u spill
for lane, pk, ph_left in [("q_E1u", [17, 23], 3), ("q_W1u", [17, 23], 3), ("q_N1u", [14, 20], 1), ("q_S1u", [14, 20], 1)]:
    j = names.index(lane)
    mm = free & (X[:, j] * CAP_U >= 4)
    if mm.sum():
        print(lane, "spill at free decisions:", mm.sum(), " pockets density mean", X[mm][:, pk].mean(0).round(2),
              " net picks", Counter(net[mm]).most_common(), " green", Counter(green[mm]).most_common())
# time course of a spill episode (ep 3) around the worst state
i0 = np.where((ep == 3))[0]
k = i0[np.argmax(up[i0].sum(1))]
for i in range(k - 12, k + 8):
    print(t[i], "g", green[i], "tg", tg[i], "net", net[i], "ph", ph[i].astype(int), "up", up[i].round(0).astype(int),
          "since", since[i].astype(int), "od", overdue[i].astype(int))
