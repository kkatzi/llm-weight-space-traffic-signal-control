"""General imputation of the dead W2 detector, folded into layer 1 by construction (no fitting).

    q_W2_hat = sum_j aq[j] * x[j] + cq
    d_W2_hat = sum_j ad[j] * x[j] + cd

The layer-1 pre-activation gets W1[:,11]*q_W2_hat + W1[:,23]*d_W2_hat, so for every input j used:
    W1[:,j] += aq[j]*W1[:,11] + ad[j]*W1[:,23],   b1 += cq*W1[:,11] + cd*W1[:,23].
Each input column used costs 64 changed parameters, the constant another 64.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools

W0 = tools.load("start_weights.npz")
C11, C23 = W0["W1"][:, 11].copy(), W0["W1"][:, 23].copy()


def fold(aq, ad, cq=0.0, cd=0.0, w=None):
    """aq, ad: dicts {input index: coefficient}."""
    w = {k: v.copy() for k, v in (W0 if w is None else w).items()}
    for j in set(aq) | set(ad):
        assert j not in (11, 23)
        w["W1"][:, j] += aq.get(j, 0.0) * C11 + ad.get(j, 0.0) * C23
    if cq or cd:
        w["b1"] += cq * C11 + cd * C23
    return w


def imputed(aq, ad, cq, cd, X):
    X = np.atleast_2d(X)
    q = cq + sum(a * X[:, j] for j, a in aq.items())
    d = cd + sum(a * X[:, j] for j, a in ad.items())
    return q, d
