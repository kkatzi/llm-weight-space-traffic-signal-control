import numpy as np, sys
sys.path.insert(0, '.')
import tools
exec(open('work/probe1.py').read().split('np.set_printoptions')[0])
ud = {k: 1.5 for k in ['N0u','N1u','S0u','S1u']}; ud.update({k: 2.0 for k in ['E0u','E1u','W0u','W1u']})
PH = {0: ['N0','N1','S0','S1'], 1: ['N2','S2'], 2: ['E0','E1','W0','W1'], 3: ['E2','W2']}
# for pairs of candidate phases (a, b), with per-lane halting na, nb (0..3), which picked
for cur in range(4):
    cands = [p for p in range(4) if p != cur]
    for i in range(3):
        for j in range(i+1, 3):
            a, b = cands[i], cands[j]
            M = np.empty((4, 4), dtype=object)
            for na in range(4):
                for nb in range(4):
                    h = {l: na for l in PH[a]}; h.update({l: nb for l in PH[b]})
                    x = state(cur, halting=h, updens=ud)
                    M[na, nb] = int(tools.greedy(w, x)[0])
            print(f'cur={cur}: rows per-lane halting at phase {a} ({len(PH[a])} lanes), cols at phase {b} ({len(PH[b])} lanes)')
            for na in range(4): print('    ', ' '.join(str(v) for v in M[na]))
