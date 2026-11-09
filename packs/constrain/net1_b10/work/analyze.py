"""Statistics of the recorded run of the network under the rules (states.npz). No fitting."""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools

w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64)
act = S["act"].astype(int)
ep = S["episode"]
ts = S["time_s"]
R = S["rule_state"]
since = R[:, 0:4]
legal = R[:, 4:8] > 0.5
over = R[:, 8:12] > 0.5
halt = R[:, 12:16]
tg = R[:, 16]
green = R[:, 17].astype(int)
L = tools.forward(w, X)
net = L.argmax(1)
print("network argmax == recorded act:", (net == act).mean())

N = len(X)
illegal = ~legal[np.arange(N), act]
print("decisions", N, " illegal requests", illegal.sum())

# classify illegal requests
want_switch = act != green
nlegal = legal.sum(1)
cls = {}
for i in np.where(illegal)[0]:
    g = green[i]
    minG = 15 if g in (0, 2) else 10
    if act[i] == g:
        k = "keep_blocked(maxgreen/overdue)"
        if over[i].any():
            k = "keep_blocked_overdue"
        elif tg[i] > 45:
            k = "keep_blocked_maxgreen"
    else:
        if tg[i] < minG:
            k = "switch_blocked_mingreen"
        elif over[i].any():
            k = "switch_to_nonoverdue_blocked"
        else:
            k = "switch_other"
    cls[k] = cls.get(k, 0) + 1
print("illegal by type:", cls)

# t_green values observed
print("t_green values:", np.unique(tg, return_counts=True))

# what shield picks: legal phase with most halting vehicles; compare to best-legal-by-logit
def shield_pick(i):
    cand = np.where(legal[i])[0]
    return cand[np.argmax(halt[i, cand])]

def logit_pick(i):
    cand = np.where(legal[i])[0]
    return cand[np.argmax(L[i, cand])]

diff = 0
n_multi = 0
for i in np.where(illegal)[0]:
    if legal[i].sum() > 1:
        n_multi += 1
        if shield_pick(i) != logit_pick(i):
            diff += 1
print("illegal with >1 legal option:", n_multi, " shield != best-logit:", diff)

# overdue statistics
print("states with any overdue:", over.any(1).sum())
print("overdue per phase:", over.sum(0))
# max since_served for phases with halting>0
print("since_served quantiles for non-green phases with halting>0:")
mask = (halt > 0) & (np.arange(4)[None, :] != green[:, None])
print(np.percentile(since[mask], [50, 90, 95, 99, 100]))

# Green durations (completed greens) per phase under the recorded run
durs = {p: [] for p in range(4)}
for e in np.unique(ep):
    idx = np.where(ep == e)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if green[b] != green[a]:
            durs[green[a]].append(tg[a] + 5 - 3)  # approx: green lasted until switch (yellow 3 s)
for p in range(4):
    d = np.array(durs[p])
    if len(d):
        print("phase", p, "n", len(d), "green dur pct", np.percentile(d, [10, 25, 50, 75, 90, 100]))

# demand per episode: count of cycles
for e in np.unique(ep):
    idx = ep == e
    print("ep", e, "decisions", idx.sum(), "illegal", illegal[idx].sum(), "overdue states", over[idx].any(1).sum(),
          "max t_green", tg[idx].max())
