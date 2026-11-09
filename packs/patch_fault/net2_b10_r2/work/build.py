"""Build repaired weights by folding an imputation of the dead W2 inputs into the first layer.

The detector of W2 reports q_W2 = x[11] = 0 and d_W2 = x[23] = 0. We replace them by estimates that are linear in
working inputs:
    q_W2_hat = a0 + sum_j a[j] * x[j]
    d_W2_hat = c0 + sum_j c[j] * x[j]
Because the first layer is linear before tanh, W1[:,11]*q_W2_hat + W1[:,23]*d_W2_hat can be folded exactly into the
columns of the proxy inputs j and into b1:
    W1[:, j] += a[j] * W1[:, 11] + c[j] * W1[:, 23]
    b1       += a0   * W1[:, 11] + c0   * W1[:, 23]
Each proxy column costs 64 changed parameters, the bias 64.

usage: python work/build.py <name> '<json spec>'   e.g.  '{"q": {"5": 1}, "d": {"17": 1}}'
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools

HERE = os.path.dirname(os.path.abspath(__file__))
Q, D = 11, 23


def build(spec, w0=None):
    w0 = tools.load("start_weights.npz") if w0 is None else w0
    w = {k: v.copy() for k, v in w0.items()}
    q = {int(k): float(v) for k, v in spec.get("q", {}).items()}
    d = {int(k): float(v) for k, v in spec.get("d", {}).items()}
    for j in sorted(set(q) | set(d)):
        if j < 0:  # bias
            continue
        w["W1"][:, j] = w0["W1"][:, j] + q.get(j, 0.0) * w0["W1"][:, Q] + d.get(j, 0.0) * w0["W1"][:, D]
    if -1 in q or -1 in d:
        w["b1"] = w0["b1"] + q.get(-1, 0.0) * w0["W1"][:, Q] + d.get(-1, 0.0) * w0["W1"][:, D]
    return w


def impute(spec, X):
    X = np.array(X, dtype=np.float64, copy=True)
    q = {int(k): float(v) for k, v in spec.get("q", {}).items()}
    d = {int(k): float(v) for k, v in spec.get("d", {}).items()}
    qh = sum(v * (X[:, j] if j >= 0 else 1.0) for j, v in q.items())
    dh = sum(v * (X[:, j] if j >= 0 else 1.0) for j, v in d.items())
    X[:, Q] = qh
    X[:, D] = dh
    return X


if __name__ == "__main__":
    name, spec = sys.argv[1], json.loads(sys.argv[2])
    w0 = tools.load("start_weights.npz")
    w = build(spec, w0)
    # exactness check: edited net on dead-detector inputs == original net on imputed inputs
    X = tools.states()["obs"].astype(np.float64)
    assert np.allclose(X[:, [Q, D]], 0)
    err = np.abs(tools.forward(w, X) - tools.forward(w0, impute(spec, X))).max()
    es = tools.edit_size(w, w0)
    print("max logit error vs imputed original:", err, "edit size:", es)
    assert es["changed"] <= 371
    path = os.path.join(HERE, name + ".npz")
    tools.save(w, path)
    # float32 round trip check
    w32 = tools.load(path)
    print("edit size after float32 save:", tools.edit_size(w32, w0))
    with open(os.path.join(HERE, name + ".json"), "w") as f:
        json.dump(spec, f)
    print("saved", path)
