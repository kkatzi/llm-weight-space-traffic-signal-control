"""probe.py: lone-vehicle starvation probe (forward passes on synthetic states).
For each stop-line lane, put one halted car (q=d=0.335) in an otherwise empty junction and follow the
network's decisions in a crude phase/timer model with 5 s steps (switch honoured only if green >= 10 s,
3 s yellow, the car is cleared once its phase has been green >= 5 s). Report lanes never served."""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import tools
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
NM = tools.names()

def run(w, lane, p0, tg0, steps=120, visible=True):
    p, tg = p0, tg0
    for k in range(steps):
        x = np.zeros(46)
        if visible or lane != 11:
            x[lane] = 0.335; x[12 + lane] = 0.335
        if lane == 11:      # failed detector: invisible
            x[11] = 0; x[23] = 0
        x[40 + p] = 1; x[44] = float(tg >= 10); x[45] = min(tg / 60, 1)
        a = int(tools.greedy(w, x)[0])
        if lane in SERVE[p] and tg >= 5:
            return k * 5
        if a != p and tg >= 10:
            p, tg = a, 2
        else:
            tg += 5
    return None

def report(w, label):
    bad = []
    for lane in range(12):
        for p0 in range(4):
            for tg0 in [2, 15, 30, 45, 60]:
                r = run(w, lane, p0, tg0)
                if r is None:
                    bad.append((NM[lane], p0, tg0))
    print(label, 'starved (lane, start phase, start green s):', bad)
    return bad

if __name__ == '__main__':
    from build import cand, W0
    report(W0, 'start')
    for nm in sys.argv[1:]:
        report(cand(nm), nm)
