"""Second round of rule comparisons on recorded states (rules only, descriptive)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools
from build import P, score_matrix
S = tools.states(); X = S["obs"].astype(np.float64); C = S["controller"]; A = S["act"]
names = tools.names(); ix = {n: i for i, n in enumerate(names)}
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
np.set_printoptions(precision=3, suppress=True, linewidth=160)
V = {"V1 (WM.5 WUM.25 K1.5)": dict(P),
     "T1 (WM1 WUM.5 K1.5)": dict(P, WM=1.0, WUM=0.5, K=1.5),
     "T2 (WM1 WUM.5 K2)": dict(P, WM=1.0, WUM=0.5, K=2.0),
     "T3 (WM1 WUM.4 K1.5)": dict(P, WM=1.0, WUM=0.4, K=1.5),
     "halting only": dict(P, WM=0.0, WUM=0.0, K=0.5)}
for c in (1, 0):
    m = ok & (C == c); cur = ph[m]
    print("states of", S["controller_names"][c])
    ref = None
    for k, Pv in V.items():
        Wsc, b = score_matrix(Pv); d = (X[m] @ Wsc.T + b).argmax(1)
        ref = d if ref is None else ref
        # full pocket handling: states where an E/W pocket is full and E1u/W1u has halting vehicles, EW_through green
        print(f"  {k:24s} keep {np.mean(d == cur):.3f} choice {np.bincount(d, minlength=4) / len(d)} agree-rec {np.mean(d == A[m]):.3f} agree-V1 {np.mean(d == ref):.3f}")
# spill situations
for k, Pv in V.items():
    Wsc, b = score_matrix(Pv); d = (X @ Wsc.T + b).argmax(1)
    out = []
    for ap, thr, lft in (("E", 2, 3), ("W", 2, 3), ("N", 0, 1), ("S", 0, 1)):
        mm = ok & (ph == thr) & (X[:, ix[f"q_{ap}2"]] > .99) & (X[:, ix[f"q_{ap}1u"]] > 0)
        out.append(f"{ap}: n {mm.sum()} ->left {np.mean(d[mm] == lft):.2f} keep {np.mean(d[mm] == thr):.2f}")
    print(f"{k:24s} through green, own pocket full + lane-1 upstream queue:", "; ".join(out))
