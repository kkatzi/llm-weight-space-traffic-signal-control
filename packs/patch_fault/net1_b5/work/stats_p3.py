"""E-side evidence: during phase 3 (EW_left green), does 'halting on E1u while E1 is not full' mean E2 still has a queue?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
St, X = load_states(); c = St["controller"]; ep = St["episode"]
ok = X[:, 44] > 0.5; ph = X[:, 40:44].argmax(1)
good = (c > 0) | (ep < 2)
w2 = tools.load(os.path.join(HERE, "work", "v2.npz"))
Q = np.round(X[:, 27] * 256 / 7.5).astype(int); D1 = np.round(X[:, 16] * 22.4 / 7.5).astype(int); E2 = np.round(X[:, 17] * 3).astype(int)
qE2 = np.round(X[:, 5] * 3).astype(int)
for dmax in [0, 1, 2, 3]:
    for qmin in [1, 2]:
        m = good & (ph == 3) & ok & (Q >= qmin) & (D1 <= dmax)
        if m.sum():
            print(f"phase3 ok, q_E1u>={qmin}, d_E1<={dmax}: n={m.sum():4d}  d_E2 dist {np.bincount(E2[m], minlength=4) / m.sum()}  q_E2 dist {np.bincount(qE2[m], minlength=4) / m.sum()}")
m = good & (ph == 3) & ok
print("all phase3 ok states n", m.sum(), "d_E2 dist", np.bincount(E2[m], minlength=4) / m.sum())
# how does the network end phase 3 on the E side: decisions when phase3, E2 empty-ish but E1u halting
m = good & (ph == 3) & ok & (Q >= 1) & (E2 <= 1)
print("phase3, q_E1u>=1, d_E2<=1car: n", m.sum(), "v2 picks", np.bincount(tools.greedy(w2, X[m]), minlength=4))
