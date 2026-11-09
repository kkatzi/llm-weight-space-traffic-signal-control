"""Build networks that feed an estimate of the dead W2 detector into the first layer.

The dead inputs x[11] (q_W2) and x[23] (d_W2) enter the first layer only through W1[:, 11] and W1[:, 23]. If we
estimate them as linear functions of live inputs,
    q_W2_hat = sum_j cq[j] * x[j],     d_W2_hat = sum_j cd[j] * x[j],
then by linearity the network that sees the estimate is obtained by
    W1[:, j] += cq[j] * W1[:, 11] + cd[j] * W1[:, 23]   for every j used.
Each live column j used costs 64 changed parameters.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools  # noqa: E402

W0 = tools.load("start_weights.npz")
IDX = {n: i for i, n in enumerate(tools.names())}


def build(cq, cd, base=W0):
    """cq, cd: dicts {input name: coefficient} for q_W2_hat and d_W2_hat."""
    w = {k: v.copy() for k, v in base.items()}
    for name in set(cq) | set(cd):
        j = IDX[name]
        w["W1"][:, j] += cq.get(name, 0.0) * base["W1"][:, 11] + cd.get(name, 0.0) * base["W1"][:, 23]
    return w


def impute_x(X, cq, cd):
    """The same estimate applied to observations (for checks: forward(W0, impute_x(X)) == forward(build(...), X))."""
    X = np.array(X, dtype=np.float64, copy=True)
    X[:, 11] = sum(c * X[:, IDX[n]] for n, c in cq.items())
    X[:, 23] = sum(c * X[:, IDX[n]] for n, c in cd.items())
    return X
