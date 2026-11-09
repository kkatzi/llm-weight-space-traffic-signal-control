"""Chain shapes with the steep part at low scores and a slow tail (forward passes only)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools
from build import P, build, rule_scores
def chain(Pv, Sg):
    al = np.broadcast_to(np.asarray(Pv["ALPHA"], float), (len(Pv["MU"]),))
    h1 = np.stack([np.tanh(a * (Sg - mu)) for a, mu in zip(al, Pv["MU"])]).sum(0)
    return Pv["GAMMA"] * np.stack([np.tanh(Pv["BETA"] * h1 + c) for c in Pv["C2"]]).sum(0)
Sg = np.array([0, 1, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50.])
S = tools.states(); X = S["obs"].astype(np.float64); ok = X[:, 44] > 0.5
r = rule_scores(P, X).argmax(1)
base = dict(ALPHA=(0.25, 0.22, 0.18, 0.12, 0.07), MU=(0.0, 4.0, 9.0, 16.0, 27.0))
out = []
for beta in (0.3, 0.36):
    for c2 in ((0.0, 0.6), (0.3, 0.9), (0.5, 1.1)):
        for gam in (3.0, 3.5, 4.0):
            Pv = dict(P, **base, BETA=beta, C2=c2, GAMMA=gam)
            w, _ = build(Pv); t = tools.trainability(w)
            g = chain(Pv, Sg); sl = np.diff(g) / np.diff(Sg)
            out.append((beta, c2, gam, t["median_max_prob"], t["share_max_prob_above_0.99"], t["max_abs_logit"], sl[[0, 2, 5, 8, 10, 11]]))
for o in out:
    print(f"beta {o[0]} c2 {o[1]} gamma {o[2]}: med {o[3]:.3f} >.99 {o[4]:.3f} maxlog {o[5]:.2f} slopes@0,2-4,8-10,15-20,25-30,30-40 {o[6].round(3)}")
