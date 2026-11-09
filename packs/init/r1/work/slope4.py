"""Layer-1-shaped chains with a near-linear layer 2 (forward passes only)."""
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
shapes = {"L1": dict(ALPHA=(0.3, 0.2, 0.12, 0.06), MU=(2.0, 7.0, 14.0, 26.0)),
          "L2": dict(ALPHA=(0.3, 0.2, 0.12, 0.06), MU=(1.0, 6.0, 13.0, 26.0)),
          "L3": dict(ALPHA=(0.35, 0.2, 0.12, 0.06), MU=(1.0, 6.0, 13.0, 26.0))}
for nm, sh in shapes.items():
    for beta in (0.2,):
        for gam in (4.5, 5.0, 5.5, 6.0):
            Pv = dict(P, **sh, BETA=beta, C2=(-0.2, 0.2), GAMMA=gam)
            w, _ = build(Pv); t = tools.trainability(w)
            g = chain(Pv, Sg); sl = np.diff(g) / np.diff(Sg)
            print(f"{nm} beta {beta} gamma {gam}: med {t['median_max_prob']:.3f} >.99 {t['share_max_prob_above_0.99']:.3f} maxlog {t['max_abs_logit']:.2f}"
                  f" sat1 {t['sat_l1']:.4f} net==rule {np.mean(tools.greedy(w, X) == r):.4f} slopes {sl.round(2)}")
