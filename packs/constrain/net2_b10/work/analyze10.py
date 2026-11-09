"""After a forced overdue insertion (network wanted X, rules served q), what does the network do at the next free
decision? Does it return to X, and if not, how long does X then wait and with how many halting vehicles?"""
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
forced = can_end & ~legal[np.arange(N), green] & overdue.any(1) & ~legal[np.arange(N), net]
rows = []
for i in np.where(forced)[0]:
    want = net[i]
    # served phase = green at i+1
    if i + 1 >= N or ep[i + 1] != ep[i]:
        continue
    q = green[i + 1]
    j = i + 1
    while j < N and ep[j] == ep[i] and green[j] == q and not can_end[j]:
        j += 1
    if j >= N or ep[j] != ep[i] or green[j] != q:
        continue
    nxt = net[j]
    # when is `want` next served?
    k = j
    while k < N and ep[k] == ep[i] and green[k] != want:
        k += 1
    wait = t[k] - t[i] if k < N and ep[k] == ep[i] else -1
    rows.append((int(green[i]), int(want), int(q), int(nxt), bool(legal[j].sum() > 1), int(ph[j, want]),
                 int(since[j, want]), int(wait), round(float(lg[j, nxt] - lg[j, want]), 2)))
ret = [r for r in rows if r[3] == r[1]]
nret = [r for r in rows if r[3] != r[1]]
print("forced insertions:", len(rows), " network returns to X:", len(ret), " goes elsewhere:", len(nret))
print("(prev green, X wanted, q forced, next pick, free?, halting X, since X, s until X served, logit gap pick-X)")
for r in nret:
    print(r)
print("wait until X served, when returning:", np.mean([r[7] for r in ret]).round(1), " when not:", np.mean([r[7] for r in nret]).round(1))
# Cue effect: at free decisions ending phase 1 (NS left), how often does the network pick 0 vs its halting
free = can_end & ~overdue.any(1)
for c in range(4):
    for p in range(4):
        if p == c:
            continue
        m = free & (green == c) & (ph[:, p] >= 7)
        if m.sum() >= 5:
            print(f"green {c}, phase {p} has >=7 halting: n={m.sum()}, network picks {p}: {(net[m] == p).mean():.2f}, picks: {Counter(net[m]).most_common()}")
