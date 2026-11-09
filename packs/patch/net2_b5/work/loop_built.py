import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
import loopcheck2
from loopcheck import LANE_PHASE
for f in sys.argv[1:]:
    w = tools.load(f)
    times = []; never = 0; tot = 0
    for nl in [1, 2, 3]:
        for lanes in itertools.combinations(range(12), nl):
            for nveh in [1, 2, 3]:
                for p0 in range(4):
                    for tg0 in [2, 12]:
                        lw = [l for l in lanes if LANE_PHASE[l] != p0]
                        if not lw: continue
                        tot += 1
                        r = loopcheck2.run(w, lw, nveh, p0, tg0, steps=60)
                        if r is None: never += 1
                        else: times.append(r)
    print('%s: configs %d never %d mean decisions %.2f p90 %.0f max %d' % (f, tot, never, np.mean(times), np.quantile(times, .9), max(times)))
