"""Frozen-observation phase-machine check (no traffic model): one car waits on one lane and nothing else is in the
network. Step the controller's own decision logic every 5 s (min green 10 s, 3 s yellow, new green is 2 s old at the
next decision) and report how long until the car's lane gets green. Lane 11 (W2) is invisible (x stays 0)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
SERVES = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}

def time_to_serve(w, lane, v, p0, g0=12, horizon=600):
    p, g = p0, g0
    for t in range(0, horizon, 5):
        if lane in SERVES[p] and t > 0:
            return t
        x = np.zeros(46); x[40+p] = 1; x[44] = 1.0 if g >= 10 else 0.0; x[45] = min(g/60, 1)
        if lane != 11:
            x[lane] = v; x[12+lane] = v
        a = int(tools.greedy(w, x)[0])
        if a != p and g >= 10:
            p, g = a, 2
        else:
            g += 5
    return horizon

def report(w, vals=(0.33,)):
    n = tools.names(); worst = {}
    for lane in range(12):
        ts = [time_to_serve(w, lane, v, p0) for v in vals for p0 in range(4) if lane not in SERVES[p0]]
        worst[n[lane]] = max(ts)
    return worst
