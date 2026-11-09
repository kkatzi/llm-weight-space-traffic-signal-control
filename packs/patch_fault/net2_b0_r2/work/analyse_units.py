"""Rank layer-1 units by how much the (copy-imputed) network's honoured decisions change when the unit is
replaced by its mean activation (mean compensated through b2). Analysis only; writes nothing."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools  # noqa: E402

w0 = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64)
ok = X[:, 44] > 0.5

# copy imputation: x11 := x5 (q_E2), x23 := x17 (d_E2)
w = {k: v.copy() for k, v in w0.items()}
w["W1"][:, 5] += w0["W1"][:, 11]
w["W1"][:, 17] += w0["W1"][:, 23]

h1 = np.tanh(X @ w["W1"].T + w["b1"])
base = (np.tanh(h1 @ w["W2"].T + w["b2"]) @ w["W3"].T + w["b3"]).argmax(1)
res = []
for j in range(64):
    h = h1.copy()
    h[:, j] = h1[:, j].mean()
    g = (np.tanh(h @ w["W2"].T + w["b2"]) @ w["W3"].T + w["b3"]).argmax(1)
    res.append(((g[ok] != base[ok]).mean(), j))
res.sort()
for r, j in res[:12]:
    print(f"unit {j:2d}: honoured decisions changed {r:.4f}, mean {h1[:, j].mean():+.2f}, std {h1[:, j].std():.2f}")
print("worst", res[-3:])
