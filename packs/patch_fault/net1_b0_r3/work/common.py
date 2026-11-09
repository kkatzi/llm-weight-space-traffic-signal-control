"""Shared helpers for the W2-detector patch (run from the pack root or work/)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import tools

NAMES = tools.names()
IDX = {n: i for i, n in enumerate(NAMES)}
S = tools.states()
X, A, EP, T, C = S["obs"].astype(np.float64), S["act"], S["episode"], S["time_s"], S["controller"]
W0 = tools.load(os.path.join(ROOT, "start_weights.npz"))


def copyw(w):
    return {k: v.copy() for k, v in w.items()}


def impute(x, a_q=0.0, b_q=1.0, a_d=0.0, b_d=1.0):
    """Insert a hypothetical W2 reading into observations (for analysis only)."""
    x = np.array(x, dtype=np.float64, copy=True)
    x[..., 11] = np.clip(a_q + b_q * x[..., 5], 0, 1)
    x[..., 23] = np.clip(a_d + b_d * x[..., 17], 0, 1)
    return x
