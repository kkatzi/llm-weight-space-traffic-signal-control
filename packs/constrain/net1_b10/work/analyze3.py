"""Where are the network's switch decisions close calls, and how do the rules interact with them? Descriptive only."""
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
ming = np.where(np.isin(green, [0, 2]), 15, 10)
sw = (tg >= ming)
order = np.argsort(-L, 1)
top, sec = order[:, 0], order[:, 1]
gap = L[np.arange(N), top] - L[np.arange(N), sec]

# 1) overdue-soon events: at a switch decision, choose q; would some other p (halting>0) reach 90 s by the time q may end?
dur = lambda q: 20 if q in (0, 2) else 15
n_ev = 0
n_sec = 0
gaps = []
for i in np.where(sw & ~over.any(1))[0]:
    q = top[i]
    if q == green[i]:
        continue
    soon = [p for p in range(4) if p != q and p != green[i] and halt[i, p] > 0 and since[i, p] + dur(q) >= 90]
    if soon:
        n_ev += 1
        p = max(soon, key=lambda p: since[i, p])
        gaps.append(L[i, q] - L[i, p])
        n_sec += sec[i] == p
print("switch decisions where the chosen green would let another phase become overdue:", n_ev,
      " of which that phase is the 2nd choice:", n_sec)
print("logit gap top - soon-overdue phase pct:", np.percentile(gaps, [10, 25, 50, 75, 90]) if gaps else None)

# 2) forced (overdue) decisions: what network wanted vs forced; gap
forced = sw & over.any(1) & ~legal[np.arange(N), act]
print("forced decisions:", forced.sum())
fg = []
for i in np.where(forced)[0]:
    p = np.where(legal[i])[0]
    fg.append(L[i, act[i]] - L[i, p].max())
print("logit gap wanted - forced pct:", np.percentile(fg, [10, 25, 50, 75, 90]))

# 3) episodes timeline: overdue count per 15-min block
blocks = (ts // 900).clip(0, 4)
for b in range(5):
    m = blocks == b
    print("block", b, "states", m.sum(), "overdue states", over[m].any(1).sum(),
          "mean halting total", halt[m].sum(1).mean().round(1),
          "mean up_queue", X[m, 24:32].mean().round(3), "pocket full share(E2,W2)",
          (X[m][:, [5, 11]] >= 0.99).mean().round(2), "(N2,S2)", (X[m][:, [2, 8]] >= 0.99).mean().round(2))

# 4) close calls at switch decisions: top through vs others
m = sw & ~over.any(1)
for thr in (0.5, 1.0, 2.0):
    mm = m & (gap < thr)
    print(f"gap<{thr}: {mm.sum()} decisions; top through: {np.isin(top[mm], [0, 2]).sum()}; "
          f"top through & sec left: {(np.isin(top[mm], [0, 2]) & np.isin(sec[mm], [1, 3])).sum()}")

# 5) halting of chosen through vs the best other at switch decisions, by halting of chosen
for p in (0, 2):
    mm = m & (top == p)
    hp = halt[mm, p]
    print("through", p, "chosen with halting <=2:", (hp <= 2).sum(), "of", mm.sum())
