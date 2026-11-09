"""Candidate H imputation (3 columns: d_E2, phase3, bias = 192 parameters).

    q_W2_hat = 0.5*d_E2 + 0.18*(1 - phase3)
    d_W2_hat = 0.5*d_E2 + 0.23*(1 - phase3)

0.5 = shrink the mirror half-way (opposite-pocket correlation is only ~0.3-0.5; the N/S analog gives a slope of
~0.45 per vehicle).  The constants make the mean of the imputation outside phase 3 equal the mean of the E2 pocket
outside phase 3 (start-network states: q .407, d .457):  .407 - .5*.457 = .18,  .457 - .5*.457 = .23.
In phase 3 (the pocket's own green) no constant: the pocket is draining.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, tools
from impute import fold, W0

A, CQ, CD = 0.5, 0.18, 0.23


def build(a=A, cq=CQ, cd=CD):
    return fold({17: a, 43: -cq}, {17: a, 43: -cd}, cq, cd)


if __name__ == "__main__":
    import reach
    S = tools.states(); X = S["obs"].astype(float); ok = X[:, 44] > 0.5
    a0 = tools.greedy(W0, X)
    for args in [(0.5, .18, .23), (1.0, 0, 0), (0.5, .25, .25), (0.7, .15, .2)]:
        w = build(*args)
        g = tools.greedy(w, X)
        print(args, tools.edit_size(w, W0)["changed"], "share", np.round(np.bincount(g[ok], minlength=4) / ok.sum(), 3),
              "differs %.3f" % (g[ok] != a0[ok]).mean())
        reach.report(w, "   ")
