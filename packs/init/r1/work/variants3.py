"""Third round: candidate final rules vs halting-only (descriptive)."""
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
base = dict(P)
V = {"halting only": dict(base, WM=0.0, WUM=0.0, K=0.5, LP=1.0),
     "V1": dict(base, WM=0.5, WUM=0.25, K=1.5, LP=1.0),
     "F1 WM1 WUM.25 LP1.25 K.75": dict(base, WM=1.0, WUM=0.25, LP=1.25, K=0.75),
     "F2 WM1 WUM.25 LP1 K.75": dict(base, WM=1.0, WUM=0.25, LP=1.0, K=0.75),
     "F3 WM.5 WUM.25 LP1.25 K.75": dict(base, WM=0.5, WUM=0.25, LP=1.25, K=0.75),
     "F4 WM1 WUM.35 LP1.25 K.75": dict(base, WM=1.0, WUM=0.35, LP=1.25, K=0.75)}
ho = None
for k, Pv in V.items():
    Wsc, b = score_matrix(Pv); d = (X @ Wsc.T + b).argmax(1)
    ho = d if ho is None else ho
    out = []
    for c in (1, 0):
        m = ok & (C == c)
        out.append(f"{S['controller_names'][c][:3]}: keep {np.mean(d[m] == ph[m]):.3f} choice {np.bincount(d[m], minlength=4) / m.sum()} agree-HO {np.mean(d[m] == ho[m]):.3f}")
    sp = []
    for ap, thr, lft in (("E", 2, 3), ("W", 2, 3), ("N", 0, 1), ("S", 0, 1)):
        mm = ok & (ph == thr) & (X[:, ix[f"q_{ap}2"]] > .99) & (X[:, ix[f"q_{ap}1u"]] > 0)
        sp.append(f"{np.mean(d[mm] == lft):.2f}")
    print(f"{k:28s}", " | ".join(out), "| spill->left E,W,N,S", ",".join(sp))
