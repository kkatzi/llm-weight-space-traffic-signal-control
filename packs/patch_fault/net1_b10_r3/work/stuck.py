"""Count synthetic near-idle states (one car halted on one visible stop-line lane, ok=1) where the network keeps a phase
that does not serve that lane, i.e. would strand the car."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
SERVES = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}

def stuck(w, tgs=(0.2, 0.5, 1.0), vals=(0.33, 0.67)):
    n = tools.names(); out = []
    for L in range(11):
        for v in vals:
            for p in range(4):
                for tg in tgs:
                    x = np.zeros(46); x[40+p] = 1; x[44] = 1; x[45] = tg; x[L] = v; x[12+L] = v
                    if tools.greedy(w, x)[0] == p and L not in SERVES[p]:
                        out.append((n[L], v, p, tg))
    return out
