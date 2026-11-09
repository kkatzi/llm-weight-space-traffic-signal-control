"""Lone-vehicle starvation probe: empty junction, one vehicle halting on one stop-line lane (or moving), current phase p
with ok=1 and various green ages. Does the net switch to the phase that serves it?"""
import numpy as np, sys
sys.path.insert(0, '.')
import tools
W = tools.load(sys.argv[1] if len(sys.argv) > 1 else 'start_weights.npz')
capS = 22.4/7.5; capU = 256/7.5
SERVE = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 0, 7: 0, 8: 1, 9: 2, 10: 2, 11: 3}   # stop-line lane -> serving phase
lanes = tools.names()[:12]
bad = []
for n in [1, 2]:
    for lane in range(12):
        sp = SERVE[lane]
        for cur in range(4):
            if cur == sp: continue
            for tg in [12, 17, 22, 27, 32, 42, 60]:
                x = np.zeros(46); x[40+cur] = 1; x[44] = 1; x[45] = min(tg/60, 1)
                x[lane] = n/capS; x[12+lane] = n/capS
                L = tools.forward(W, x)[0]; pick = int(L.argmax())
                if pick != sp:
                    bad.append((n, lanes[lane], cur, tg, pick, np.round(L, 2)))
print('cases where the net does NOT switch to serve a lone halting vehicle (n veh, lane, cur, t_green, pick, logits):')
for b in bad: print('  ', b)
print('total bad', len(bad))
