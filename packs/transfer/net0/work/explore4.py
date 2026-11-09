"""How does the transferred net respond to spillback on each upstream lane (q and d raised together)?"""
import numpy as np
from common import *
np.set_printoptions(precision=3, suppress=True, linewidth=200)
S = tools.states(); X = S['obs'].astype(float)
ok = X[:, 52] > .5
Xo = X[ok]
w = build(mapping())
base = tools.greedy(w, Xo)
print('base pick dist (ok states)', np.bincount(base, minlength=4) / len(base))
for lane in ['N0u', 'N1u', 'S0u', 'S1u', 'E0u', 'E1u', 'E2u', 'W0u', 'W1u', 'W2u']:
    row = []
    for add in [0.1, 0.2, 0.4]:
        Xp = Xo.copy()
        for k in ['q', 'd']:
            Xp[:, IW[f'{k}_{lane}']] = np.clip(Xp[:, IW[f'{k}_{lane}']] + add, 0, 1)
        a = tools.greedy(w, Xp)
        row.append(np.bincount(a, minlength=4) / len(a))
    print(f'{lane:4s} +.1', row[0], ' +.2', row[1], ' +.4', row[2])
print('stop-line lanes full (q=d=1):')
for lane in ['N0', 'N1', 'N2', 'S0', 'S1', 'S2', 'E0', 'E1', 'E2', 'E3', 'W0', 'W1', 'W2', 'W3']:
    Xp = Xo.copy()
    for k in ['q', 'd']:
        Xp[:, IW[f'{k}_{lane}']] = 1.0
    a = tools.greedy(w, Xp)
    print(f'{lane:4s}', np.bincount(a, minlength=4) / len(a))
