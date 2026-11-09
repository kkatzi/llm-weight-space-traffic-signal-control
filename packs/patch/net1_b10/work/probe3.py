import numpy as np, sys
sys.path.insert(0, '.')
import tools
exec(open('work/probe1.py').read().split('np.set_printoptions')[0])
np.set_printoptions(precision=2, suppress=True, linewidth=160)
ud = {k: 1.5 for k in ['N0u','N1u','S0u','S1u']}; ud.update({k: 2.0 for k in ['E0u','E1u','W0u','W1u']})
PH = {0: ['N0','N1','S0','S1'], 1: ['N2','S2'], 2: ['E0','E1','W0','W1'], 3: ['E2','W2']}
UP = {0: ['N0u','N1u','S0u','S1u'], 1: ['N1u','S1u'], 2: ['E0u','E1u','W0u','W1u'], 3: ['E1u','W1u']}
# cur=2: phase 0 full stop-line (3/lane) + upstream queue u0 per lane; phase 3 full (3/lane) + upstream on E1u/W1u u3
for cur, a, b in [(2, 0, 3), (0, 2, 1), (1, 0, 2), (3, 0, 2), (2, 0, 1), (0, 2, 3)]:
    print(f'cur={cur}: rows upstream queue per lane at {a}, cols upstream per lane at {b}; both stop-lines full')
    for ua in [0, 2, 5, 10]:
        row = []
        for ub in [0, 2, 5, 10]:
            h = {l: 3 for l in PH[a] + PH[b]}
            u = {l: ua for l in UP[a]}
            for l in UP[b]: u[l] = max(u.get(l, 0), ub)
            x = state(cur, halting=h, up=u, updens=ud)
            # updens overwritten: ensure upstream density at least queue
            for l in u: x[ix['d_'+l]] = max(x[ix['d_'+l]], min(u[l]/capU, 1))
            row.append(int(tools.greedy(w, x)[0]))
        print('    ', ua, row)
