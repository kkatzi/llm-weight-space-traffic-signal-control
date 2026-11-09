"""Phase sequences, overdue events and since_served at switch decisions in the recorded run."""
import os, sys
import numpy as np
from collections import Counter
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools

S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
act = S["act"].astype(int)
rs = S["rule_state"]
since = rs[:, 0:4]; legal = rs[:, 4:8] > 0.5; overdue = rs[:, 8:12] > 0.5; ph = rs[:, 12:16]
tg = rs[:, 16].astype(int); green = rs[:, 17].astype(int)
ep = S["episode"]; t = S["time_s"]
lg = tools.forward(w, X)
N = len(act)

# green sequence per episode, with green durations
trans = Counter(); dur = {p: [] for p in range(4)}
for e in np.unique(ep):
    idx = np.where(ep == e)[0]
    seq = []
    for i in idx:
        if tg[i] == 2 or (i == idx[0]):
            seq.append(green[i])
    for a, b in zip(seq[:-1], seq[1:]):
        trans[(a, b)] += 1
    print("ep", e, "n greens", len(seq), "steps", len(idx), "end time", t[idx[-1]], "seq head", "".join(map(str, seq[:60])))
print("transitions:", sorted(trans.items()))
# green durations: last tg before a new green
for i in range(N - 1):
    if ep[i + 1] == ep[i] and tg[i + 1] == 2 and tg[i] >= 10:
        dur[green[i]].append(tg[i])
for p in range(4):
    print("phase", p, "ended at tg:", Counter(dur[p]).most_common())

# overdue events
od = overdue.any(1)
print("\noverdue states:", od.sum())
print("overdue phase counts:", overdue.sum(0))
print("overdue states by tg:", Counter(tg[od]).most_common())
# overdue at a switch decision (current green may end) -> forced
dec = (legal.sum(1) >= 1) & od & (~legal[np.arange(N), green])
print("forced (current illegal, overdue):", dec.sum())
# network's request vs overdue phase at those
cnt = Counter()
for i in np.where(dec)[0]:
    L = np.where(legal[i])[0]
    sh = L[np.argmax(ph[i, L])]
    cnt[(int(green[i]), int(act[i]), int(sh))] += 1
print("(green, net request, served) at forced overdue:", sorted(cnt.items(), key=lambda kv: -kv[1]))
print("halting on forced overdue phase:", Counter(ph[dec][np.arange(dec.sum()), [np.where(legal[i])[0][np.argmax(ph[i, np.where(legal[i])[0]])] for i in np.where(dec)[0]]].astype(int)).most_common())
# network request equal to overdue phase anyway?
same = 0
for i in np.where(dec)[0]:
    same += legal[i, act[i]]
print("network request already legal at forced decisions:", same, "of", dec.sum())

# since_served at switching decisions (tg>=12 for left, >=17 for through)
can_end = np.where(np.isin(green, [0, 2]), tg >= 15, tg >= 10)
print("\nswitch-possible decisions:", can_end.sum())
oth = since.copy(); oth[np.arange(N), green] = -1
mx = oth.max(1)
print("max since_served (other phases) at switch-possible decisions: pct",
      np.percentile(mx[can_end], [10, 25, 50, 75, 90, 95, 99]))
# which phase has largest since at those decisions & halting
for p in range(4):
    m = can_end & (green != p)
    print("phase", p, "since_served pct at decisions:", np.percentile(since[m, p], [25, 50, 75, 90, 99]).round(0),
          " halting mean:", ph[m, p].mean().round(2))
# per-phase halting overall
print("mean phase_halting:", ph.mean(0).round(2))
