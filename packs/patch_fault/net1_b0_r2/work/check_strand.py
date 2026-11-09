"""Stranding check: one halting car on a stop-line lane (or one car on an upstream lane), nothing else visible,
current green has lasted >= 60 s (t_green = 1, the state repeats unchanged). Does the network keep a phase that
does not serve that car? Compared for start, patch-1-only and final networks."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools

w0 = tools.load("start_weights.npz")
wi = {k: v.copy() for k, v in w0.items()}
wi["W1"][:, 5] += w0["W1"][:, 11]; wi["W1"][:, 17] += w0["W1"][:, 23]
wf = tools.load(os.path.join(HERE, "weights.npz"))
names = tools.names()
serve = {"N0": {0}, "N1": {0}, "N2": {1}, "E0": {2}, "E1": {2}, "E2": {3}, "S0": {0}, "S1": {0}, "S2": {1},
         "W0": {2}, "W1": {2}}
up_serve = {"N0u": {0}, "N1u": {0, 1}, "E0u": {2}, "E1u": {2, 3}, "S0u": {0}, "S1u": {0, 1}, "W0u": {2}, "W1u": {2, 3}}
res = {"start": [], "patch1": [], "final": []}
for nets in [("start", w0), ("patch1", wi), ("final", wf)]:
    nm, w = nets
    stuck = []
    for li, ln in enumerate(["N0", "N1", "N2", "E0", "E1", "E2", "S0", "S1", "S2", "W0", "W1"]):
        for p in range(4):
            if p in serve[ln]:
                continue
            x = np.zeros(46); x[40 + p] = 1; x[44] = 1; x[45] = 1.0; x[li] = 0.3348; x[12 + li] = 0.3348
            if tools.greedy(w, x)[0] == p:
                stuck.append("%s@ph%d" % (ln, p))
    for ui, ln in enumerate(["N0u", "N1u", "E0u", "E1u", "S0u", "S1u", "W0u", "W1u"]):
        for p in range(4):
            if p in up_serve[ln]:
                continue
            x = np.zeros(46); x[40 + p] = 1; x[44] = 1; x[45] = 1.0; x[24 + ui] = 0.0293; x[32 + ui] = 0.0293
            if tools.greedy(w, x)[0] == p:
                stuck.append("%s@ph%d" % (ln, p))
    print("%-7s stuck single-car states: %2d  %s" % (nm, len(stuck), " ".join(stuck)))
