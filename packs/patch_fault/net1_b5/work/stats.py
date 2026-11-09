"""Data statistics: long greens, and how the E-side pocket E2 relates to upstream E1u (E side has all detectors)."""
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
import tools
np.set_printoptions(precision=3, suppress=True, linewidth=220)
S = tools.states(); X = S["obs"].astype(float); c = S["controller"]; ep = S["episode"]
good = (c > 0) | (ep < 2)
ph = X[:, 40:44].argmax(1); tg = X[:, 45]
vis = [i for i in list(range(12, 23)) + list(range(32, 40))]
dsum = X[:, vis].sum(1)
for nm, m in [("start ep0-1", (c == 0) & (ep < 2)), ("fixed", c == 1), ("actuated", c == 2)]:
    print(nm, "t_green>=0.9 & phase!=3:", ((tg >= 0.9) & (ph != 3) & m).sum(), "/", m.sum(),
          "  dsum<0.5:", ((dsum < 0.5) & m).sum(), " max t_green", tg[m].max())
    for th in [0.3, 0.45, 0.6, 0.75, 0.9]:
        print(f"   tg>={th}: by phase", [int(((tg >= th) & (ph == p) & m).sum()) for p in range(4)])
# E2 vs E1u
m = good
for cond_name, cond in [("q_E1u==0", X[:, 27] == 0), ("q_E1u>0", X[:, 27] > 0), ("q_E1u>=0.1", X[:, 27] >= 0.1)]:
    mm = m & cond
    print(cond_name, mm.sum(), "q_E2 dist", np.bincount(np.round(X[mm, 5] * 3).astype(int), minlength=4) / max(mm.sum(), 1))
print("corr q_E2 vs q_W1u-ish proxies on E side:", np.corrcoef(X[m, 5], X[m, 27])[0, 1], np.corrcoef(X[m, 5], X[m, 35])[0, 1])
# under phase 2 green (E straight served), E1u queue with E1 empty
mm = m & (ph == 2) & (X[:, 27] > 0) & (X[:, 16] < 0.34)
print("phase2 & q_E1u>0 & d_E1<=1car:", mm.sum(), "q_E2 dist", np.bincount(np.round(X[mm, 5] * 3).astype(int), minlength=4) / max(mm.sum(), 1))
# W side in fixed vs E side: W1u queue when phase 2
for side, q1u, d1 in [("E", 27, 16), ("W", 31, 22)]:
    mm = (c == 1) & (ph == 2)
    print(side, "fixed, phase2: mean q_1u", X[mm, q1u].mean(), "mean d_1", X[mm, d1].mean())
