"""Compare decisions of rule variants on recorded ok=1 states (rules only, no network, no fitting)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools
from build import P, score_matrix

S = tools.states(); X = S["obs"].astype(np.float64); C = S["controller"]; A = S["act"]
names = tools.names(); ix = {n: i for i, n in enumerate(names)}
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
CAP_U = 256 / 7.5

def upstream_moving(X):
    """moving vehicles on each phase's upstream lanes (lane-1 split 50/50)"""
    m = np.zeros((len(X), 4))
    for p, lanes in {0: [("N0u", 1), ("S0u", 1), ("N1u", .5), ("S1u", .5)], 1: [("N1u", .5), ("S1u", .5)],
                     2: [("E0u", 1), ("W0u", 1), ("E1u", .5), ("W1u", .5)], 3: [("E1u", .5), ("W1u", .5)]}.items():
        for l, s in lanes:
            m[:, p] += s * CAP_U * (X[:, ix["d_" + l]] - X[:, ix["q_" + l]])
    return m

def rule(Pv, X, red_extra=0.0):
    Wsc, b = score_matrix(Pv)
    s = X @ Wsc.T + b
    if red_extra:
        s += red_extra * upstream_moving(X) * (1 - X[:, 40:44])
    return s

variants = {
    "V1 base": (dict(P), 0.0),
    "V2 WM=.8": (dict(P, WM=0.8), 0.0),
    "V3 WM=.8 red+.35": (dict(P, WM=0.8, WUM=0.2), 0.35),
    "V4 K=2.5": (dict(P, K=2.5), 0.0),
    "V5 halting only": (dict(P, WM=0.0, WUM=0.0, K=0.5), 0.0),
}
m = ok & (C == 1)
dec = {k: rule(Pv, X[m], r).argmax(1) for k, (Pv, r) in variants.items()}
cur = ph[m]
for k, d in dec.items():
    print(f"{k:20s} keep {np.mean(d == cur):.3f} choice {np.bincount(d, minlength=4) / len(d)} agree-actuated {np.mean(d == A[m]):.3f}"
          f" agree-V1 {np.mean(d == dec['V1 base']):.3f}")
m2 = ok & (C == 0)
print("fixed-time states:")
for k, (Pv, r) in variants.items():
    d = rule(Pv, X[m2], r).argmax(1)
    print(f"{k:20s} keep {np.mean(d == ph[m2]):.3f} choice {np.bincount(d, minlength=4) / len(d)}")
