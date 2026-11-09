"""Descriptive statistics of the recorded states (no fitting)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools

S = tools.states()
X = S["obs"].astype(np.float64)
A = S["act"]
E = S["episode"]
T = S["time_s"]
C = S["controller"]
names = tools.names()
np.set_printoptions(precision=3, suppress=True, linewidth=160)

print("controllers", S["controller_names"])
for c in range(2):
    m = C == c
    print("controller", c, "n", m.sum(), "episodes", np.unique(E[m]))
    print(" mean obs:")
    for i, n in enumerate(names):
        print(f"  {n:7s} mean {X[m, i].mean():.3f}  q90 {np.quantile(X[m, i], 0.9):.3f}  max {X[m, i].max():.3f}  frac>0 {(X[m, i] > 0).mean():.2f}")

# episode lengths, phase durations
for c in range(2):
    m = C == c
    for e in np.unique(E[m]):
        mm = m & (E == e)
        t = T[mm]
        ph = X[mm, 40:44].argmax(1)
        # green durations: count consecutive decisions with same phase
        changes = np.flatnonzero(np.diff(ph) != 0)
        segs = np.diff(np.r_[-1, changes, len(ph) - 1])
        seq = ph[np.r_[changes, len(ph) - 1]]
        print(f"ctrl {c} ep {e}: n {mm.sum()} t_end {t.max()} switches {len(changes)} mean seg decisions {segs.mean():.2f}")
        for p in range(4):
            sp = segs[seq == p]
            print(f"   phase {p}: count {len(sp)} mean decisions {sp.mean() if len(sp) else 0:.2f} max {sp.max() if len(sp) else 0}")
