"""Frozen check with a full stop-line lane plus halted vehicles on its upstream lane (drain rules inactive)."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
from loopcheck import LANE_PHASE
UPL = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 5, 8: 5, 9: 6, 10: 7, 11: 7}
names = tools.names()
def run(w, lane, nup, p0, tg0, steps=40):
    phase, tg = p0, tg0
    for k in range(steps):
        x = np.zeros(46)
        x[lane] = 1; x[12 + lane] = 1
        u = UPL[lane]; x[24 + u] = nup / 34.13; x[32 + u] = nup / 34.13
        x[40 + phase] = 1; x[44] = float(tg >= 10); x[45] = min(tg / 60, 1)
        a = int(tools.greedy(w, x)[0])
        if a != phase and tg >= 10:
            phase, tg = a, 2
            if LANE_PHASE[lane] == phase: return k
        else: tg += 5
    return None
for f in sys.argv[1:]:
    w = tools.load(f); bad = []
    for lane in range(12):
        for nup in [1, 3, 8]:
            for p0 in range(4):
                if LANE_PHASE[lane] == p0: continue
                for tg0 in [2, 12]:
                    r = run(w, lane, nup, p0, tg0)
                    if r is None or r > 8: bad.append((names[lane], nup, p0, tg0, r))
    print(f, 'full-lane+upstream configs not served within ~40 s:', len(bad), bad[:12])
