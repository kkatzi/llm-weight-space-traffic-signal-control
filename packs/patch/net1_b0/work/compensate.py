import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, tools
from build import build, ideal_logits
w0 = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); ok = X[:, 44] > .5; ctl = S["controller"]
cfg = dict(alpha=0.75, M=2.5, G=5.0)
K, J = (31, 27, 59), (25, 35, 41)
w = build(w0, X_mean=X[ok], k_units=K, j_units=J, **cfg)
gi = ideal_logits(w0, X, **cfg).argmax(1)
h1, _ = tools.hidden(w0, X[ok])
shift = sum(w0["W2"][:, k] * h1[:, k].mean() for k in K)   # lost mean input to each layer-2 unit
shift[list(J)] = 0
order = np.argsort(-np.abs(shift))
budget = 371 - tools.edit_size(w, w0)["changed"]
print("budget left", budget, "largest shifts", np.round(shift[order[:budget]], 3))
for nb in [0, 5, 10, budget]:
    w2 = {k: v.copy() for k, v in w.items()}
    for i in order[:nb]: w2["b2"][i] += shift[i]
    g = tools.greedy(w2, X)
    print(nb, "edit", tools.edit_size(w2, w0)["changed"], "built!=ideal (ok states):", round((g[ok] != gi[ok]).mean(), 4),
          "ctl0:", round((g[ok & (ctl==0)] != gi[ok & (ctl==0)]).mean(), 4))
