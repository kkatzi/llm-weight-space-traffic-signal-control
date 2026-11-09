"""Trainability as a function of the read-out scale GAMMA and layer-2 gain BETA (forward passes only)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools
from build import P, build, rule_scores
S = tools.states(); X = S["obs"].astype(np.float64); ok = X[:, 44] > 0.5
r = rule_scores(P, X).argmax(1)
for beta in (0.35, 0.45, 0.6):
    for g in (3.5, 4.0, 4.5, 5.0, 5.5):
        w, _ = build(dict(P, GAMMA=g, BETA=beta))
        t = tools.trainability(w); to = tools.trainability(w, X[ok])
        print(f"beta {beta} gamma {g}: med {t['median_max_prob']:.3f} >.99 {t['share_max_prob_above_0.99']:.3f} maxlog {t['max_abs_logit']:.2f}"
              f" gap {t['median_logit_gap']:.2f} | ok-only med {to['median_max_prob']:.3f} >.99 {to['share_max_prob_above_0.99']:.3f}"
              f" | net==rule {np.mean(tools.greedy(w, X) == r):.4f}")
