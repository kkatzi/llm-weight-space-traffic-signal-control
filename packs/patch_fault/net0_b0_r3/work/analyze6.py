"""Analogue test (input level) of the 'spillback indicator' idea on pockets with a working detector.

Pretend pocket P2 failed. Candidate imputations (fixed formulas, nothing fitted):
  zero          : P2 = 0 (the fault)
  copy          : P2 = opposite pocket (same phase serves both)
  ind           : P2 = 1 if >=1 halted car on P1u and P's left phase not green, else 0
  ind|copy      : P2 = 1 if indicator else opposite pocket
  ind|mean      : P2 = 1 if indicator else the not-served mean
Score: agreement of the start network's honoured decisions with those it makes on the true input.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools

w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64)
ctrl = S["controller"]
nm = tools.names()
ix = {n: i for i, n in enumerate(nm)}
ok = X[:, 44] > 0.5
ph = X[:, 40:44].argmax(1)
a_true = tools.greedy(w, X)
OPP = {"N": "S", "S": "N", "E": "W"}
LP = {"N": 1, "S": 1, "E": 3}
one_car = 1 / (256 / 7.5)

for P in ["N", "S", "E"]:
    q, d = ix[f"q_{P}2"], ix[f"d_{P}2"]
    o = OPP[P]
    qo, do = ix[f"q_{o}2"], ix[f"d_{o}2"]
    ind = (X[:, ix[f"q_{P}1u"]] > 0.5 * one_car) & (ph != LP[P])
    ns = ph != LP[P]
    mq, md = X[ns & ~ind, q].mean(), X[ns & ~ind, d].mean()
    cands = {
        "zero": (0 * X[:, q], 0 * X[:, d]),
        "copy": (X[:, qo], X[:, do]),
        "ind": (ind * 1.0, ind * 1.0),
        "ind|copy": (np.where(ind, 1.0, X[:, qo]), np.where(ind, 1.0, X[:, do])),
        "ind|mean": (np.where(ind, 1.0, mq * ns), np.where(ind, 1.0, md * ns)),
        "max(copy,ind)+": (np.maximum(X[:, qo], ind), np.maximum(X[:, do], ind)),
    }
    print(f"\n### pretend {P}2 failed (opposite {o}2{' also failed' if o == 'W' else ''}); indicator fires on "
          f"{ind.mean():.3f} of states")
    for name, (qq, dd) in cands.items():
        Z = X.copy(); Z[:, q] = qq; Z[:, d] = dd
        a = tools.greedy(w, Z)
        row = [f"{(a[ok & (ctrl == c)] == a_true[ok & (ctrl == c)]).mean():.3f}" for c in range(3)]
        mi = ok & ind
        print(f"  {name:16s} agree by src {row} all {(a[ok] == a_true[ok]).mean():.3f}  "
              f"on indicator states {(a[mi] == a_true[mi]).mean():.3f}  "
              f"on non-ind {(a[ok & ~ind] == a_true[ok & ~ind]).mean():.3f}")
