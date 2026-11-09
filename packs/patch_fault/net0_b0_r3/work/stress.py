"""Synthetic stress checks: sweep the number of halted cars on W1u in honoured phase-2 / phase-3 / phase-0 states
and look at the final network's choices and logit range (forward passes only)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np
import tools
from build import Q_W1U

w0 = tools.load("start_weights.npz")
w = tools.load(os.path.join(HERE, "weights.npz"))
wc = tools.load(os.path.join(HERE, "weights_copy_only.npz"))
X = tools.states()["obs"].astype(np.float64)
ok = X[:, 44] > 0.5
ph = X[:, 40:44].argmax(1)
car = 1 / (256 / 7.5)
for p in [0, 2, 3]:
    base = X[ok & (ph == p)]
    print(f"phase {p} honoured states (n={len(base)}): share of choices as halted cars on W1u varies")
    for n in [0, 1, 2, 5, 10, 20, 34]:
        Z = base.copy(); Z[:, Q_W1U] = min(1.0, n * car)
        Z[:, 39] = np.maximum(Z[:, 39], Z[:, Q_W1U])   # density >= queue
        lg = tools.forward(w, Z)
        a = lg.argmax(1)
        ac = tools.greedy(wc, Z)
        print(f"   n={n:2d}: final {np.bincount(a, minlength=4) / len(a)}  copy-only "
              f"{np.bincount(ac, minlength=4) / len(a)}  max|logit| {np.abs(lg).max():.2f}")
