"""Logit-vs-score chain and trainability for candidate chain shapes (forward passes only)."""
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
cfgs = {
 "C1": dict(P, ALPHA=(0.2, 0.2, 0.2, 0.12, 0.08), MU=(1.0, 6.0, 11.0, 18.0, 28.0), BETA=0.33, C2=(-0.3, 0.3), GAMMA=4.0),
 "C2": dict(P, ALPHA=(0.2, 0.2, 0.2, 0.12, 0.08), MU=(1.0, 6.0, 11.0, 18.0, 28.0), BETA=0.30, C2=(-0.3, 0.3), GAMMA=4.0),
 "C3": dict(P, ALPHA=(0.22, 0.2, 0.18, 0.12, 0.08), MU=(1.0, 6.0, 11.0, 18.0, 28.0), BETA=0.3, C2=(-0.25, 0.25), GAMMA=4.0),
 "C4": dict(P, ALPHA=(0.2, 0.2, 0.2, 0.12, 0.08), MU=(1.0, 6.0, 11.0, 18.0, 28.0), BETA=0.28, C2=(-0.3, 0.3), GAMMA=3.6),
}
for k, Pv in cfgs.items():
    g = chain(Pv, Sg)
    w, _ = build(Pv)
    t = tools.trainability(w); to = tools.trainability(w, X[ok])
    print(k, "logit(S):", dict(zip(Sg.astype(int).tolist(), g.round(2).tolist())))
    print("   slope:", (np.diff(g) / np.diff(Sg)).round(3))
    print(f"   med {t['median_max_prob']:.3f} >.99 {t['share_max_prob_above_0.99']:.3f} maxlog {t['max_abs_logit']:.2f} gap {t['median_logit_gap']:.2f}"
          f" sat1 {t['sat_l1']:.4f} sat2 {t['sat_l2']:.4f} dead {t['dead_l1']},{t['dead_l2']} | ok-only med {to['median_max_prob']:.3f} >.99 {to['share_max_prob_above_0.99']:.3f}"
          f" | net==rule {np.mean(tools.greedy(w, X) == r):.4f}")
