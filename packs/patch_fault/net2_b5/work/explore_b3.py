import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, tools
from impute import fold, W0
from drain_table import obs
import reach
S = tools.states(); X = S["obs"].astype(float); ok = X[:, 44] > 0.5
P3 = 43
mq, md = .407, .457
variants = {
    "half_p3zero": ({5: .5, P3: -.5 * mq}, {17: .5, P3: -.5 * md}, .5 * mq, .5 * md),
    "half_phase": ({5: .5, P3: .5 * (.056 - mq)}, {17: .5, P3: .5 * (.219 - md)}, .5 * mq, .5 * md),
}
for name, (aq, ad, cq, cd) in variants.items():
    base = fold(aq, ad, cq, cd)
    g0 = tools.greedy(base, X)
    for d2 in [0, 0.25, 0.5, 0.75, 1.0]:
        w = {k: v.copy() for k, v in base.items()}; w["b3"][2] += d2
        g = tools.greedy(w, X)
        print(f"{name} b3[2]+={d2}: flips vs base {(g[ok] != g0[ok]).mean():.3f}", end="  ")
        reach.report(w, "")
# logits along W0x1 cycle
w = fold(*variants["half_p3zero"])
for p, a in [(0, 12), (1, 12), (1, 32), (3, 12), (3, 47), (3, 62)]:
    print(p, a, np.round(tools.forward(w, obs({"W0": 1}, p, a))[0], 2))
