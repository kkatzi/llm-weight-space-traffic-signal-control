import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
import loopcheck2, rulesim
from loopcheck import LANE_PHASE
w0 = tools.load('start_weights.npz')
orig = tools.greedy

def stats(greedy_fn, label):
    loopcheck2.tools.greedy = greedy_fn
    times = []; never = 0; tot = 0
    for nl in [1, 2, 3]:
        for lanes in itertools.combinations(range(12), nl):
            for nveh in [1, 2]:
                for p0 in range(4):
                    lw = [l for l in lanes if LANE_PHASE[l] != p0]
                    if not lw: continue
                    tot += 1
                    r = loopcheck2.run(w0, lw, nveh, p0, steps=60)
                    if r is None: never += 1
                    else: times.append(r)
    print('%-28s configs %d never %d  mean decisions-to-serve-all %.2f  p90 %.0f' % (label, tot, never, np.mean(times), np.quantile(times, .9)))

stats(orig, 'original')
for rules in [('R1',), ('E0', 'E2'), ('R1', 'E0', 'E2'), ('R1', 'E0', 'E2', 'E3'), ('R1', 'E0', 'E1', 'E2', 'E3')]:
    stats(rulesim.make_greedy(rules), str(rules))
