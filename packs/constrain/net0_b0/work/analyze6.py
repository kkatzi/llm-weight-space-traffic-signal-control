"""How often would a conservative 'serve the soon-overdue phase first' swap trigger in the recorded run?
Pure counting for a rule whose numbers were chosen by reasoning (Smith's rule margin 1.5)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools

S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
R = S["rule_state"]
ss, legal, over, ph, tg, gr = R[:, 0:4], R[:, 4:8] > 0.5, R[:, 8:12] > 0.5, R[:, 12:16], R[:, 16], R[:, 17].astype(int)
act = S["act"].astype(int)
ep, t = S["episode"], S["time_s"]
L = tools.forward(w, X)
np.set_printoptions(linewidth=200, precision=2, suppress=True)

D = lambda k: 20.0 if k % 2 == 0 else 15.0
for kappa in [1.0, 1.5, 2.0]:
    n = 0
    rows = []
    for i in np.where(legal.sum(1) == 4)[0]:
        c, g = act[i], gr[i]
        if c == g:
            continue
        best = None
        for p in range(4):
            if p in (c, g) or ph[i, p] <= 0:
                continue
            if ss[i, p] + D(c) >= 90 and ph[i, p] / D(p) > kappa * ph[i, c] / D(c):
                best = p
        if best is not None:
            n += 1
            rows.append((ep[i], t[i], g, c, best, ph[i].copy(), ss[i].copy(), L[i].copy()))
    print(f"kappa {kappa}: triggers {n}  (per episode {n/6:.1f})")
    if kappa == 1.5:
        for r in rows:
            print("  ", r)
