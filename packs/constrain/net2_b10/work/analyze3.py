"""Effective decisions (current green may end): what the network picks, with what margin, and the halting counts."""
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
print("effective decisions:", can_end.sum(), " free of overdue:", free.sum())
print("net picks at free decisions:", Counter(net[free]).most_common())
print("net stays at free decisions:", (net[free] == green[free]).sum())
srt = np.sort(lg, 1)
gap = srt[:, -1] - srt[:, -2]
print("logit gap at free decisions pct:", np.percentile(gap[free], [10, 25, 50, 75, 90]).round(2))
# chosen phase halting vs max halting among non-current
for p in range(4):
    m = free & (net == p)
    print(f"pick {p}: n={m.sum()} halting of pick mean {ph[m, p].mean():.2f}, pct",
          np.percentile(ph[m, p], [10, 25, 50]).round(1), " gap median", np.median(gap[m]).round(2),
          " since of pick median", np.median(since[m, p]))
# how often net picks the phase with most halting (among non-current)?
oth = ph.copy(); oth[np.arange(N), green] = -1
mh = oth.argmax(1)
print("net pick == most halting non-current:", (net[free] == mh[free]).mean().round(3))
# picks of through phases with few halting vehicles
for p in [0, 2]:
    m = free & (net == p)
    print(f"through pick {p} with halting<=2:", (m & (ph[:, p] <= 2)).sum(), "of", m.sum())
# second choice
sec = np.argsort(lg, 1)[:, -2]
print("(pick, second) at free decisions:", Counter(zip(net[free], sec[free])).most_common(12))
# since_served ages of left phases when picked
print("lane-level: through-phase picks where halting small; show some")
for i in np.where(free & np.isin(net, [0, 2]) & (ph[np.arange(N), net] <= 1))[0][:15]:
    print(green[i], tg[i], net[i], ph[i].astype(int), since[i].astype(int), lg[i].round(2))
