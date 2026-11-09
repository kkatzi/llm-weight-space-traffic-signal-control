import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load('start_weights.npz')
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
def state(cur, levels, tg=12, up=0.0):
    x = np.zeros(46)
    for p in range(4):
        for l in SERVE[p]:
            if p == cur:
                x[l] = 0; x[12 + l] = levels[p] / 3
            else:
                x[l] = levels[p] / 3; x[12 + l] = levels[p] / 3
    x[32:40] = up
    x[40 + cur] = 1; x[44] = 1 if tg >= 10 else 0; x[45] = tg / 60
    return x
for cur in range(4):
    print('current phase', cur)
    for base in [0, 1, 2]:
        row = []
        for r in range(4):
            if r == cur: continue
            res = []
            for k in [0, 1, 2, 3]:
                lv = [base] * 4; lv[r] = k; lv[cur] = 1
                res.append(int(tools.greedy(w, state(cur, lv))[0]))
            row.append((r, res))
        print('   others at', base, ' (target, picks for target level 0..3):', row)
