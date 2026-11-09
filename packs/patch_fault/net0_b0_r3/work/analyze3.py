"""Analogue test of imputation ideas on pockets whose detector WORKS.

For a working pocket P (N2, S2, E2) we pretend its detector failed (q_P2 = d_P2 = 0) and compare the start
network's honoured decisions with the decisions it makes when P2 is seen correctly. Then we try a few hand-made
imputations of P2 from other observed inputs and see how much of the agreement each brings back.
Nothing is fitted: each candidate is a fixed formula.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools

np.set_printoptions(precision=3, suppress=True, linewidth=160)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64)
ctrl = S["controller"]
nm = tools.names()
ix = {n: i for i, n in enumerate(nm)}
ok = X[:, 44] > 0.5
a_true = tools.greedy(w, X)

OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}


def test(P):
    q, d = ix[f"q_{P}2"], ix[f"d_{P}2"]
    o = OPP[P]
    qo, do = ix[f"q_{o}2"], ix[f"d_{o}2"]
    q1u, d1u = ix[f"q_{P}1u"], ix[f"d_{P}1u"]
    mq, md = X[:, q].mean(), X[:, d].mean()
    cands = {
        "zero (failed)": lambda Z: (0 * Z[:, q], 0 * Z[:, d]),
        "mean const": lambda Z: (mq + 0 * Z[:, q], md + 0 * Z[:, d]),
        "copy opposite": lambda Z: (Z[:, qo], Z[:, do]),
        "half opp + half mean": lambda Z: (0.5 * Z[:, qo] + 0.5 * mq, 0.5 * Z[:, do] + 0.5 * md),
        "copy opp + 4*own d1u": lambda Z: (Z[:, qo] + 4 * Z[:, d1u], Z[:, do] + 4 * Z[:, d1u]),
        "half opp + 4*own d1u": lambda Z: (0.5 * Z[:, qo] + 4 * Z[:, d1u], 0.5 * Z[:, do] + 4 * Z[:, d1u]),
        "half opp + half mean + 2*d1u": lambda Z: (0.5 * Z[:, qo] + 0.5 * mq + 2 * Z[:, d1u],
                                                    0.5 * Z[:, do] + 0.5 * md + 2 * Z[:, d1u]),
    }
    print(f"\n### pretend {P}2 failed (opposite pocket {o}2{' is itself failed' if o == 'W' else ''})")
    for name, f in cands.items():
        Z = X.copy()
        qq, dd = f(X)
        Z[:, q], Z[:, d] = qq, dd
        a = tools.greedy(w, Z)
        row = []
        for c in range(3):
            m = ok & (ctrl == c)
            row.append(f"{(a[m] == a_true[m]).mean():.3f}")
        m3 = ok & ((a_true == 3) | (a == 3) | (a_true == 1) | (a == 1))
        print(f"  {name:32s} agree(ok) by data src {row}  all {(a[ok] == a_true[ok]).mean():.3f}"
              f"  | on left-phase-relevant {(a[m3] == a_true[m3]).mean():.3f}")


for P in ["N", "S", "E"]:
    test(P)
