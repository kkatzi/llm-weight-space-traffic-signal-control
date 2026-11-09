"""Pick the units to repurpose for the final configuration (discrete choice by forward passes; no fitting)."""
import sys, os, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, tools
from build import build, ideal_logits
w0 = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); ok = X[:, 44] > .5
Xo = X[ok]
cfg = dict(alpha=0.75, M=2.5, G=5.0)
gi = ideal_logits(w0, Xo, **cfg).argmax(1)
h1c = [31, 5, 27, 59, 61, 35, 1, 6, 42, 22, 34, 0, 4]
h2c = [25, 44, 35, 6, 54, 39, 15, 41, 46, 4]
res = []
for ks in itertools.combinations(h1c, 3):
    w = build(w0, X_mean=Xo, k_units=ks, j_units=(25, 35, 41), **cfg)
    res.append(((tools.greedy(w, Xo) != gi).mean(), ks))
res.sort(); print("top h1 triples", res[:5])
res2 = []
for ks in [r[1] for r in res[:3]]:
    for js in itertools.combinations(h2c, 3):
        w = build(w0, X_mean=Xo, k_units=ks, j_units=js, **cfg)
        res2.append(((tools.greedy(w, Xo) != gi).mean(), ks, js))
res2.sort(); print("top combos", res2[:5])
