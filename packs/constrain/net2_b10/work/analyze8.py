"""Look for costly patterns in the recorded run: low-content greens, overdue insertions per episode,
and what the network wants right after a forced overdue green."""
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
can_end = np.where(np.isin(green, [0, 2]), tg >= 15, tg >= 10)
forced = can_end & ~legal[np.arange(N), green] & overdue.any(1)
eff_forced = forced & ~legal[np.arange(N), net]
for e in np.unique(ep):
    m = ep == e
    print(f"ep {e}: forced overdue decisions {forced[m].sum()}, of which network overruled {eff_forced[m].sum()}, "
          f"mean halting {ph[m].sum(1).mean():.1f}")
# served phase content at the start of each green: halting at the switching decision
sw = np.where(can_end[:-1] & (tg[1:] == 2) & (ep[1:] == ep[:-1]))[0]
served = green[sw + 1]
h = ph[sw, served]
print("greens started:", len(sw), " with 0 halting:", (h == 0).sum(), " with 1:", (h == 1).sum(), " with 2:", (h == 2).sum())
for p in range(4):
    m = served == p
    print(f"phase {p}: started {m.sum()}, halting at start pct {np.percentile(h[m], [10, 25, 50, 75]).round(1)}")
# forced greens content
fs = np.where(forced[:-1] & (tg[1:] == 2))[0]
print("forced greens:", len(fs), "halting on served phase at start:", Counter(ph[fs, green[fs + 1]].astype(int)).most_common())
# After a forced green, what does the network ask at its first free decision? Is it the one it asked before?
agree = 0; tot = 0
for i in np.where(eff_forced)[0]:
    want = net[i]
    j = i + 1
    while j < N and ep[j] == ep[i] and not (can_end[j] and tg[j] < 30 and green[j] != green[i]):
        j += 1
    while j < N and ep[j] == ep[i] and not can_end[j]:
        j += 1
    if j < N and ep[j] == ep[i]:
        tot += 1; agree += net[j] == want
print("after forced insertion, network asks for the phase it wanted before:", agree, "of", tot)
