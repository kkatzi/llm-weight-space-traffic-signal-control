"""Choose which hidden units to repurpose: the ones whose loss disturbs the original decisions least
(forward passes on recorded states only; no weights are fitted)."""
import sys, os, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, tools
from build import build, ideal_logits
w0 = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); ok = X[:, 44] > .5
Xo = X[ok]
gi = ideal_logits(w0, Xo).argmax(1)
h1c = [31, 5, 27, 59, 61, 35, 1, 6, 42, 22, 34, 0, 4]
h2c = [25, 44, 35, 6, 54, 39, 15, 41, 46, 4]
# greedy: first pick h1 triple with fixed h2, then h2 triple
best = None
for ks in itertools.combinations(h1c, 3):
    w = build(w0, X_mean=Xo, k_units=ks, j_units=(25, 35, 41))
    e = (tools.greedy(w, Xo) != gi).mean()
    if best is None or e < best[0]: best = (e, ks)
print("best h1", best)
ks = best[1]; best2 = None
for js in itertools.combinations(h2c, 3):
    w = build(w0, X_mean=Xo, k_units=ks, j_units=js)
    e = (tools.greedy(w, Xo) != gi).mean()
    if best2 is None or e < best2[0]: best2 = (e, js)
print("best h2", best2)
