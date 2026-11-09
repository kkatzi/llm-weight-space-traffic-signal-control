"""Build repaired weights by folding an estimate of the dead W2 inputs into the first layer.

The dead inputs x[11] (q_W2) and x[23] (d_W2) enter layer 1 only through W1[:, 11] and W1[:, 23].
If we want the network to behave as if it saw
    q_hat = cq + sum_j aq[j] * x[j]      d_hat = cd + sum_j ad[j] * x[j]
then, because layer 1 is linear before the tanh, it is enough to set
    b1      += W1[:, 11] * cq    + W1[:, 23] * cd
    W1[:, j] += W1[:, 11] * aq[j] + W1[:, 23] * ad[j]      for every source input j
Each source input costs 64 changed parameters, the constant another 64. W1[:, 11] and W1[:, 23] themselves are
left alone (they multiply inputs that are always 0, so they no longer matter).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools  # noqa: E402

N = tools.names()
I = {k: i for i, k in enumerate(N)}
QW2, DW2 = I["q_W2"], I["d_W2"]


def build(cq, cd, aq, ad, w0=None):
    """aq / ad: dict input-name -> coefficient for q_hat / d_hat."""
    w0 = tools.load("start_weights.npz") if w0 is None else w0
    w = {k: v.copy() for k, v in w0.items()}
    w["b1"] = w["b1"] + w0["W1"][:, QW2] * cq + w0["W1"][:, DW2] * cd
    for name in sorted(set(aq) | set(ad)):
        j = I[name]
        w["W1"][:, j] = w0["W1"][:, j] + w0["W1"][:, QW2] * aq.get(name, 0.0) + w0["W1"][:, DW2] * ad.get(name, 0.0)
    return w


def impute(X, cq, cd, aq, ad):
    q = cq + sum(c * X[:, I[k]] for k, c in aq.items())
    d = cd + sum(c * X[:, I[k]] for k, c in ad.items())
    return q, d


DESIGNS = {
    # E2 shrunk toward a small constant, plus spill-back on W1u, plus a floor that grows with the green time
    "A": dict(cq=0.05, cd=0.10, aq={"q_E2": 0.5, "q_W1u": 3.0, "t_green": 0.4},
              ad={"d_E2": 0.5, "q_W1u": 3.0, "t_green": 0.4}),
    # same structure, leaning more on the opposite pocket E2
    "C": dict(cq=0.03, cd=0.06, aq={"q_E2": 0.8, "q_W1u": 3.0, "t_green": 0.4},
              ad={"d_E2": 0.8, "q_W1u": 3.0, "t_green": 0.4}),
}

if __name__ == "__main__":
    name = sys.argv[1]
    d = DESIGNS[name]
    w = build(**d)
    w0 = tools.load("start_weights.npz")
    print(tools.edit_size(w, w0))
    out = os.path.join(HERE, "work", f"w_{name}.npz")
    tools.save(w, out)
    print("saved", out)
