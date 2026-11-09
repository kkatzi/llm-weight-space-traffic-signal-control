from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = tools.load("../start_weights.npz")
def trace(w, lanes, counts=None, start=2, maxsteps=40):
    """Static demand; follow decisions until a phase serving a waiting lane is chosen with ok, or loop."""
    need = set(lane_phase(l) for l in lanes)
    p, tg = start, 12; path = []
    for s in range(maxsteps):
        x = make_state(lanes, p, tg, counts)
        a = int(tools.greedy(w, x)[0])
        path.append((p, tg, a))
        if tg >= 10 and a in need and a != p: return 'served', path
        if p in need: return 'served', path
        if tg >= 10 and a != p:
            p, tg = a, 2
        else:
            tg = min(tg + 5, 60) if tg < 60 else 60
            if tg == 60 and len(path) > 2 and path[-2][:2] == (p, 60): return 'STUCK', path
    return 'LOOP', path
res = {}
import itertools
nstuck = 0
for mask in range(1, 4096):
    lanes = [l for l in range(12) if mask >> l & 1]
    for c in (1, 2, 3):
        for st in range(4):
            r, path = trace(w, lanes, [c]*len(lanes), start=st)
            if r != 'served':
                nstuck += 1
                if c == 1 and len(lanes) <= 3: print(lanes, 'count', c, 'start', st, r, path[-3:])
                res.setdefault(c, []).append((tuple(lanes), st))
print('n not served', nstuck, {c: len(v) for c, v in res.items()})
from collections import Counter
for c, v in res.items():
    cnt = Counter(l for (ls, st) in v for l in ls)
    print('count', c, 'lane frequency in stuck configs', sorted(cnt.items()))
