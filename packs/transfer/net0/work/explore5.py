# Exploration script (run during design). "plain()" = lane mapping without the unstick term.
"""Stuck-green pathology: does the source net refuse to switch once t_green is large?"""
import numpy as np
from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
L2 = tools.states('states_L2.npz'); X2 = L2['obs'].astype(float)
src = tools.load('../source_weights_L2.npz')
stuck = X2[(L2['episode'] == 2) & (L2['time_s'] == 4500)][0]
print('stuck L2 state, logits by t_green (s):')
for tg in [12, 17, 22, 27, 32, 42, 52, 60]:
    x = stuck.copy(); x[45] = tg / 60
    print(tg, tools.forward(src, x)[0], tools.greedy(src, x)[0])
# all L2 ok states: decision vs t_green, holding everything else fixed
ok = (X2[:, 44] > .5) & (np.round(X2[:, 45] * 60) == 12)
Xo = X2[ok]; cur = Xo[:, 40:44].argmax(1)
print('L2 ok states at tg=12: n =', len(Xo))
for tg in [12, 17, 22, 32, 42, 60]:
    x = Xo.copy(); x[:, 45] = tg / 60
    a = tools.greedy(src, x)
    print(f'  set tg={tg:2d}: stay share {np.mean(a == cur):.3f}  pick dist {np.bincount(a, minlength=4)/len(a)}')
# same on L2W ok states with the delivered net
S = tools.states(); X = S['obs'].astype(float)
w = plain()
okw = X[:, 52] > .5
Xw = X[okw]; curw = Xw[:, 48:52].argmax(1)
print('L2W ok states: n =', len(Xw), ' t_green values:', np.unique(np.round(Xw[:, 53] * 60)))
for tg in [12, 17, 22, 32, 42, 60]:
    x = Xw.copy(); x[:, 53] = tg / 60
    a = tools.greedy(w, x)
    print(f'  set tg={tg:2d}: stay share {np.mean(a == curw):.3f}  pick dist {np.bincount(a, minlength=4)/len(a)}')
# sparse-traffic probes in L2W: one or two vehicles waiting on a red movement, current green empty
print('L2W sparse probes (current phase -> pick) at tg=12 / 60')
lanesets = {'1 car N0': ['N0'], '1 car N2': ['N2'], '1 car E3': ['E3'], '1 car W1': ['W1'], '1 car S1': ['S1'],
            'N0+N1+W3 (L2 stuck analog)': ['N0', 'N1', 'W3']}
for nm, lanes in lanesets.items():
    res = []
    for tg in (12, 60):
        r = []
        for p in range(4):
            x = np.zeros(54)
            for l in lanes:
                v = 0.391 if l[0] in 'NS' else 0.335
                x[IW['q_' + l]] = v; x[IW['d_' + l]] = v
            x[48 + p] = 1; x[52] = 1; x[53] = tg / 60
            r.append(int(tools.greedy(w, x)[0]))
        res.append(r)
    print(f'  {nm:28s} tg12 {res[0]}  tg60 {res[1]}')
