import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}   # stop-line lanes served by each phase
ONE = 1 / (22.4 / 7.5)   # one vehicle on a stop-line lane
ONEU = 1 / (256 / 7.5)   # one vehicle on an upstream lane

def make_state(lanes, phase, tg_s, counts=None):
    """lanes: list of stop-line lane indices with halting vehicles (1 each unless counts)."""
    x = np.zeros(46)
    for k, l in enumerate(lanes):
        n = 1 if counts is None else counts[k]
        x[l] = min(1, n * ONE); x[12 + l] = min(1, n * ONE)
    x[40 + phase] = 1
    x[44] = 1.0 if tg_s >= 10 else 0.0
    x[45] = min(1, tg_s / 60)
    return x

def lane_phase(l):
    for p, ls in SERVE.items():
        if l in ls: return p

# density inputs (stop-line + upstream feeding lanes) that count as vehicles for each phase
DL = {0: [12, 13, 18, 19, 32, 33, 36, 37], 1: [14, 20, 33, 37], 2: [15, 16, 21, 22, 34, 35, 38, 39], 3: [17, 23, 35, 39]}

def phase_empty(X):
    X = np.atleast_2d(X)
    return np.stack([X[:, DL[q]].sum(1) < 0.5 * ONEU for q in range(4)], 1)

def make_policy(w, vetoset=(), V=40.0):
    def pol(X):
        lg = tools.forward(w, X)
        E = phase_empty(X)
        for q in vetoset: lg[:, q] -= V * E[:, q]
        return lg.argmax(1)
    return pol

def trace(pol, lanes, counts=None, start=2, maxsteps=40):
    need = set(lane_phase(l) for l in lanes)
    p, tg = start, 12; path = []
    for s in range(maxsteps):
        x = make_state(lanes, p, tg, counts)
        a = int(pol(x)[0]); path.append((p, tg, a))
        if p in need: return 'served', path
        if tg >= 10 and a in need: return 'served', path
        if tg >= 10 and a != p: p, tg = a, 2
        else:
            if tg == 60 and len(path) > 2 and path[-2][:2] == (p, 60): return 'STUCK', path
            tg = min(tg + 5, 60)
    return 'LOOP', path

def all_traces(pol, counts=(1, 2, 3)):
    fails = []; wasted = 0
    for mask in range(1, 4096):
        lanes = [l for l in range(12) if mask >> l & 1]
        for c in counts:
            for st in range(4):
                r, path = trace(pol, lanes, [c] * len(lanes), start=st)
                if r != 'served': fails.append((tuple(lanes), c, st, r))
                else: wasted += sum(1 for (p, tg, a) in path if tg >= 10 and a != p) - 1 if len(path) > 1 else 0
    return fails, wasted
