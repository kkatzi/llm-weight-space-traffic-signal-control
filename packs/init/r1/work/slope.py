"""Logit as a function of a phase's score S along the per-phase chain (synthetic, forward only)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools
from build import P, build, rule_scores
def chain(Pv, Sg):
    h1 = np.stack([np.tanh(Pv["ALPHA"] * (Sg - mu)) for mu in Pv["MU"]]).sum(0)
    return Pv["GAMMA"] * np.stack([np.tanh(Pv["BETA"] * h1 + c) for c in Pv["C2"]]).sum(0)
Sg = np.array([0, 1, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50.])
S = tools.states(); X = S["obs"].astype(np.float64); sc = rule_scores(P, X)
print("score quantiles (all phases, recorded states) 50/90/99/max:", np.quantile(sc, [.5, .9, .99]).round(2), sc.max().round(2))
cfgs = {"cur": dict(P),
        "A": dict(P, ALPHA=0.15, MU=(0.0, 7.0, 14.0, 22.0), BETA=0.4, C2=(-0.3, 0.3), GAMMA=4.0),
        "B": dict(P, ALPHA=0.15, MU=(0.0, 6.0, 12.0, 19.0, 27.0), BETA=0.35, C2=(-0.35, 0.35), GAMMA=4.0)}
for k, Pv in cfgs.items():
    g = chain(Pv, Sg)
    print(k, "logit(S):", dict(zip(Sg.astype(int).tolist(), g.round(2).tolist())))
    print("   slope per vehicle:", (np.diff(g) / np.diff(Sg)).round(3))
