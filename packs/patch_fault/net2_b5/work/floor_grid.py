"""Small design grid (forward passes only): mirror coefficient a, floor f, and which phases get the floor."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, tools
from impute import fold, W0
import reach
S = tools.states(); X = S["obs"].astype(float); ok = X[:, 44] > 0.5
for a in [1.0, 0.5]:
    for phases in [(0, 1, 2), (0, 2), (1, 2), (2,), (0, 1), (0,), (1,)]:
        for f in [0.15, 0.25, 0.35]:
            aq = {5: a}; ad = {17: a}
            for p in phases:
                aq[40 + p] = f; ad[40 + p] = f
            w = fold(aq, ad)
            g = tools.greedy(w, X)
            sh = np.bincount(g[ok], minlength=4) / ok.sum()
            bad = reach.report(w, f"a={a} floor-phases={phases} f={f} p3share={sh[3]:.3f}", verbose=False)
