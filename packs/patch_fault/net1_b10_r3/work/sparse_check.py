"""Frozen-queue phase-machine check on random sparse states (1-3 halted cars on random stop-line lanes, W2 invisible):
each car leaves when its lane gets green; report the share of states where some car is still waiting after 300 s and
the mean time until all are served. No traffic model: just the controller's own decision logic."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
SERVES = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}

def run(w, cars, p0, g0=12, horizon=300):
    q = np.zeros(12)
    for L in cars: q[L] += 1/3
    q = np.minimum(q, 1)
    p, g = p0, g0
    for t in range(0, horizon, 5):
        for L in SERVES[p]:
            if g >= 4: q[L] = 0          # served once green has been on a few seconds
        if q.sum() == 0: return t
        x = np.zeros(46); x[40+p] = 1; x[44] = 1.0 if g >= 10 else 0.0; x[45] = min(g/60, 1)
        x[:11] = q[:11]; x[12:23] = q[:11]           # W2 (index 11 / 23) stays 0: failed detector
        a = int(tools.greedy(w, x)[0])
        if a != p and g >= 10: p, g = a, 2
        else: g += 5
    return horizon

def check(w, n=600, seed=0):
    rng = np.random.default_rng(seed); ts = []
    for _ in range(n):
        k = rng.integers(1, 4); cars = list(rng.integers(0, 12, size=k)); p0 = int(rng.integers(0, 4))
        ts.append(run(w, cars, p0))
    ts = np.array(ts)
    return {"stuck_share": float((ts >= 300).mean()), "mean_t_clear": float(ts.mean())}
