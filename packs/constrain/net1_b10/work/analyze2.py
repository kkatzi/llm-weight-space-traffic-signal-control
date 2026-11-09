"""Phase-choice statistics at decisions where a switch is possible. Descriptive only (no fitting)."""
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
N = len(X)
np.set_printoptions(precision=2, suppress=True, linewidth=150)

# Overdue mechanics: since_served when overdue first flagged
print("since_served of overdue phases:", np.unique(since[over], return_counts=True))
# legal pattern when overdue
for i in np.where(over.any(1))[0][:15]:
    print(ts[i], "green", green[i], "tg", tg[i], "since", since[i], "halt", halt[i], "legal", legal[i].astype(int),
          "over", over[i].astype(int), "act", act[i])

# switch-possible decisions: ok and t_green >= min green
ming = np.where(np.isin(green, [0, 2]), 15, 10)
sw = (tg >= ming)
print("\nswitch-possible decisions:", sw.sum())
print("act==green at those:", (act[sw] == green[sw]).sum())

# transition matrix of actual greens (from -> to) as realised
T = np.zeros((4, 4), int)
for e in np.unique(ep):
    idx = np.where(ep == e)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if green[b] != green[a]:
            T[green[a], green[b]] += 1
print("realised transitions from->to:\n", T)

# network request at switch-possible decisions vs current green
T2 = np.zeros((4, 4), int)
for i in np.where(sw)[0]:
    T2[green[i], act[i]] += 1
print("network requests at switch-possible decisions (from green -> requested):\n", T2)

# halting on chosen vs on others, at switch-possible decisions with no overdue
m = sw & ~over.any(1)
srt = np.sort(L, 1)
gap = srt[:, -1] - srt[:, -2]
print("logit gap pct at switch-possible:", np.percentile(gap[m], [10, 25, 50, 75, 90]))
ch_h = halt[np.arange(N), act]
print("halting of chosen phase pct:", np.percentile(ch_h[m], [10, 25, 50, 75, 90]))
mx_h = np.array([max(halt[i, p] for p in range(4) if p != green[i]) for i in range(N)])
print("share chosen == max-halting non-current phase:", (ch_h[m] == mx_h[m]).mean())
# by chosen phase: halting of chosen when it is through
for p in range(4):
    mm = m & (act == p)
    if mm.sum():
        print("chosen", p, "n", mm.sum(), "halt chosen pct", np.percentile(halt[mm, p], [10, 25, 50, 75]),
              " zero-halt share", (halt[mm, p] == 0).mean())

# requests at the blocked decision (t_green=12 on through) vs at 17
for g in (0, 2):
    i12 = np.where((green == g) & (tg == 12))[0]
    same = 0
    n = 0
    for i in i12:
        j = i + 1
        if j < N and ep[j] == ep[i] and tg[j] == 17 and green[j] == g:
            n += 1
            same += act[j] == act[i]
    print(f"through {g}: request at 12 equals request at 17: {same}/{n}")

# second-choice phase and its logit distance at switch-possible decisions
order = np.argsort(-L, 1)
sec = order[:, 1]
print("\nsecond choice == current green share:", (sec[m] == green[m]).mean())

# per-phase wait at time of service: since_served of the phase at the decision it gets requested & switched
ws = {p: [] for p in range(4)}
for i in np.where(sw)[0]:
    if act[i] != green[i] and legal[i, act[i]]:
        ws[act[i]].append(since[i, act[i]])
for p in range(4):
    print("phase", p, "since_served at service pct", np.percentile(ws[p], [10, 25, 50, 75, 90, 100]))

# cycle structure: phases served between consecutive services of the same phase
