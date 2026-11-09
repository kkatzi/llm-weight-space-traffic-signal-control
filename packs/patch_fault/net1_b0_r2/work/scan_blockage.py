"""Evaluate a W-side blockage unit G (phase 2 green & ok & queue in W1u & W1 lane empty -> pocket W2 is full,
98-100 % on the mirror E side). When G fires it adds a fixed layer-2 shift = mean over rotated signature states
of [a2 with W2 = full] - [a2 with W2 = E2 substitute]. Only forward passes / activation means."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools
from mirror_test import rot

S = tools.states(); X = S["obs"].astype(float); c = S["controller"]; ep = S["episode"]
w0 = tools.load("start_weights.npz"); wf = tools.load(os.path.join(HERE, "weights.npz"))
ph = X[:, 40:44].argmax(1)
sig = (c > 0) & (ph == 2) & (X[:, 44] > 0.5) & (X[:, 27] >= 0.029) & (X[:, 16] == 0)
R = rot(X[sig])

def a2(w, Z):
    return np.tanh(Z @ w["W1"].T + w["b1"]) @ w["W2"].T + w["b2"]

deltas = {}
for v in (0.0, 0.3348):
    T = R.copy(); T[:, 5] = v; T[:, 17] = v
    F = T.copy(); F[:, 11] = 1.0; F[:, 23] = 1.0          # truth: W2 full
    I = T.copy(); I[:, 11] = v; I[:, 23] = v              # what the patched net effectively sees
    deltas[v] = (a2(w0, F) - a2(w0, I)).mean(0)
    print("delta from E2=%.2f: norm %.2f" % (v, np.linalg.norm(deltas[v])))

for dv, D in deltas.items():
    for scale in (0.5, 1.0):
        line = "shift(E2=%.2f) x%.1f:" % (dv, scale)
        for v in (0.0, 0.3348, 0.6696, 1.0):
            T = R.copy(); T[:, 5] = v; T[:, 17] = v
            ref = tools.greedy(w0, np.where(np.arange(46) == 11, T, T))   # full detector sees true W2 (= rotated E2, ~full)
            Dx = T.copy(); Dx[:, 11] = 0; Dx[:, 23] = 0
            Lf = np.tanh(a2(wf, Dx) + scale * D) @ wf["W3"].T + wf["b3"]
            Lp = tools.forward(wf, Dx)
            line += "  E2=%.2f agree %.2f (was %.2f)" % (v, (Lf.argmax(1) == ref).mean(), (Lp.argmax(1) == ref).mean())
        print(line)
