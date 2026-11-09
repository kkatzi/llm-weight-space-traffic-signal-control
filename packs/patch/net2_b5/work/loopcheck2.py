"""Frozen-observation service check for 1 or 2 waiting lanes (all combos), every start phase.
A lane is 'served' when its phase gets green; the check reports configurations where some waiting lane is not
served within 60 s (12 decisions) or never."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
from loopcheck import SERVE, LANE_PHASE
names = tools.names()


def run(w, lanes, nveh, p0, tg0=12, steps=40):
    phase, tg = p0, tg0
    waiting = set(lanes); first = {}
    for k in range(steps):
        x = np.zeros(46)
        for l in waiting:
            x[l] = nveh / 2.987; x[12 + l] = nveh / 2.987
        x[40 + phase] = 1; x[44] = 1.0 if tg >= 10 else 0.0; x[45] = min(tg / 60, 1)
        a = int(tools.greedy(w, x)[0])
        if a != phase and tg >= 10:
            phase, tg = a, 2
            served = {l for l in waiting if LANE_PHASE[l] == phase}
            for l in served: first[l] = k
            waiting -= served   # served lanes clear during the 12 s green
        else:
            tg += 5
        if not waiting:
            return max(first.values())
    return None


def sweep(w, label, verbose=True):
    worst = []
    tot = 0
    for nl in [1, 2]:
        for lanes in itertools.combinations(range(12), nl):
            for nveh in [1, 2]:
                for p0 in range(4):
                    lanes_w = [l for l in lanes if LANE_PHASE[l] != p0]
                    if not lanes_w: continue
                    tot += 1
                    r = run(w, lanes_w, nveh, p0)
                    if r is None or r > 12:
                        worst.append((tuple(names[l] for l in lanes_w), nveh, p0, r))
    never = [x for x in worst if x[3] is None]
    print('%s: configs %d, slow (>60 s) %d, never served %d' % (label, tot, len(worst) - len(never), len(never)))
    if verbose:
        from collections import Counter
        cnt = Counter()
        for lanes, nveh, p0, r in never:
            for l in lanes: cnt[(l, p0)] += 1
        print('   never-served (lane, start phase) counts:', dict(cnt.most_common(20)))
    return worst


if __name__ == '__main__':
    sweep(tools.load('start_weights.npz'), 'original')
    for f in sys.argv[1:]:
        sweep(tools.load(f), f)
