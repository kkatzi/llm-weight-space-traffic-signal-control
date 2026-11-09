import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
from loopcheck import SERVE, LANE_PHASE
names = tools.names()
w = tools.load(sys.argv[1] if len(sys.argv) > 1 else 'start_weights.npz')
def seq(lane, nveh, p0, tg0=12, steps=14):
    phase, tg = p0, tg0; out = []
    for k in range(steps):
        x = np.zeros(46)
        if LANE_PHASE[lane] != phase:
            x[lane] = nveh / 2.987; x[12 + lane] = nveh / 2.987
        x[40 + phase] = 1; x[44] = 1.0 if tg >= 10 else 0.0; x[45] = min(tg / 60, 1)
        lg = tools.forward(w, x)[0]; a = int(lg.argmax())
        out.append((phase, tg, a, lg.round(1)))
        if a != phase and tg >= 10: phase, tg = a, 2
        else: tg += 5
    return out
for lane, nveh, p0 in [(9, 1, 0), (9, 1, 3), (7, 1, 2), (0, 1, 2), (9, 1, 1)]:
    print('lane', names[lane], 'veh', nveh, 'start', p0)
    for s in seq(lane, nveh, p0):
        if s[1] >= 10: print('   phase', s[0], 'tg', s[1], '-> pick', s[2], s[3])
