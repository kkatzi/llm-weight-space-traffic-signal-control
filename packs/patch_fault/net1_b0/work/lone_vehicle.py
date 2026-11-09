"""Decision-graph check on frozen sparse states: a lone halted vehicle on lane L (traffic frozen, only the
signal logic of TASK.md is stepped). Reports seconds until the phase serving L turns green (cap 600 s).
Not a traffic simulation: no vehicle moves; it only follows the controller's own phase decisions."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools, numpy as np
from probe import state, LANE_PHASE

def time_to_serve(w, lane, p0, age0=12, cap=600, visible=True):
    p, age, t = p0, age0, 0
    target = LANE_PHASE[lane]
    while t < cap:
        if p == target:
            return t
        lanes = [lane] if visible else []
        x = state(p, min(age / 60, 1), ok=int(age >= 10), lanes=lanes)
        a = int(tools.greedy(w, x[None])[0])
        if a != p and age >= 10:
            p, age = a, 2
        else:
            age += 5
        t += 5
    return None

if __name__ == '__main__':
    w0 = tools.load('start_weights.npz')
    w = tools.load(os.path.join(HERE, 'weights.npz'))
    lanes = [l for l in LANE_PHASE]
    for name, ww in [('start (faulty input)', w0), ('delivered', w)]:
        tot, stuck = [], 0
        rows = []
        for l in lanes:
            vis = l != 'W2'
            r = []
            for p0 in range(4):
                if p0 == LANE_PHASE[l]:
                    continue
                tt = time_to_serve(ww, l, p0, visible=vis)
                r.append(tt)
                if tt is None: stuck += 1
                else: tot.append(tt)
            rows.append((l, r))
        print('==', name, ': stuck (>600 s) cases', stuck, 'of', sum(len(r) for _, r in rows), '| mean wait of served %.0f s' % np.mean(tot))
        for l, r in rows:
            print('   ', l, r)
