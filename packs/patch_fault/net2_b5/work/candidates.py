"""Imputation candidates for the dead W2 detector, folded into the first layer by construction.

If x11 (q_W2) is replaced by  a*x5 + cq  and x23 (d_W2) by  a*x17 + cd, the first-layer pre-activation becomes
    W1 @ x + b1 + W1[:,11]*(a*x5 + cq) + W1[:,23]*(a*x17 + cd)
so  W1[:,5] += a*W1[:,11],  W1[:,17] += a*W1[:,23],  b1 += cq*W1[:,11] + cd*W1[:,23].
x11 and x23 are always 0 in deployment, so W1[:,11] and W1[:,23] are left untouched (they multiply 0).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools

W0 = tools.load("start_weights.npz")


def impute(a, cq, cd, w=None):
    w = {k: v.copy() for k, v in (W0 if w is None else w).items()}
    c11, c23 = W0["W1"][:, 11].copy(), W0["W1"][:, 23].copy()
    if a != 0:
        w["W1"][:, 5] += a * c11
        w["W1"][:, 17] += a * c23
    if cq != 0 or cd != 0:
        w["b1"] += cq * c11 + cd * c23
    return w


if __name__ == "__main__":
    S = tools.states(); X = S["obs"].astype(float); c = S["controller"]
    ok = X[:, 44] > 0.5
    mq, md = X[c == 0, 5].mean(), X[c == 0, 17].mean()      # E2 means under the start network
    print("E2 means (start_network states): q %.3f d %.3f" % (mq, md))
    a0 = tools.greedy(W0, X)
    for name, (a, cq, cd) in {"faulty": (0, 0, 0), "mirror": (1, 0, 0), "half": (0.5, 0.5 * mq, 0.5 * md),
                              "const": (0, mq, md)}.items():
        w = impute(a, cq, cd)
        g = tools.greedy(w, X)
        print(f"{name:7s} changed {tools.edit_size(w, W0)['changed']:3d}  honoured share",
              np.round(np.bincount(g[ok], minlength=4) / ok.sum(), 3),
              " differs from faulty %.3f" % (g[ok] != a0[ok]).mean())
