"""For each forced-overdue switch, look back at the previous free (all-legal) switch decision:
what did the network prefer there, and how close was the soon-to-be-overdue phase in logits?
Also: at free switch decisions, how much demand does the chosen phase have?"""
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
np.set_printoptions(precision=1, suppress=True, linewidth=200)

forced = np.where(overdue.any(1) & ~legal[np.arange(N), green])[0]
gaps = []
for i in forced:
    p = int(np.where(overdue[i])[0][0])
    j = i - 1
    while j >= 0 and ep[j] == ep[i] and not free[j]:
        j -= 1
    if j < 0 or ep[j] != ep[i]:
        continue
    gap = lg[j].max() - lg[j, p]
    gaps.append(gap)
    print(f"ep{ep[i]} forced t{ts[i]} p{p} net_then{net[i]} lg{lg[i]} ph{ph[i].astype(int)} | prev free t{ts[j]} g{green[j]} "
          f"net{net[j]} lg{lg[j]} ss_p{since[j,p]:.0f} ph{ph[j].astype(int)} gap{gap:.1f}")
gaps = np.array(gaps)
print("gap at previous free decision: quantiles", np.percentile(gaps, [10, 25, 50, 75, 90]).round(2))

# all-legal decisions: chosen phase demand
print("\nfree decisions:", free.sum())
ch = net[free]
pf = ph[free]
print("chosen phase halting at free decisions, through:", Counter(pf[np.arange(len(ch)), ch][ch % 2 == 0].astype(int).tolist()))
print("chosen phase halting at free decisions, left:", Counter(pf[np.arange(len(ch)), ch][ch % 2 == 1].astype(int).tolist()))
srt = np.sort(lg[free], 1)
print("top1-top2 logit gap at free decisions quantiles:", np.percentile(srt[:, -1] - srt[:, -2], [5, 10, 25, 50, 75]).round(2))
# is the network's choice the max-halting phase?
oth = pf.copy(); oth[np.arange(len(ch)), green[free]] = -1
print("net choice == max halting (excluding current):", (oth.argmax(1) == ch).mean())
