import numpy as np
from common import *
np.set_printoptions(precision=3, suppress=True, linewidth=200)
S = tools.states(); X = S['obs'].astype(float); ctrl = S['controller']
L2 = tools.states('states_L2.npz'); X2 = L2['obs'].astype(float)
src = tools.load('../source_weights_L2.npz')
ok = X[:, IW['ok']] > .5
cur = X[:, 48:52].argmax(1)

def summarize(name, w):
    a = tools.greedy(w, X)
    p = tools.probs(w, X)
    sw = ok & (a != cur)
    agree = [(a[ok & (ctrl == c)] == S['act'][ok & (ctrl == c)]).mean() for c in [0, 1]]
    print(f'{name:28s} act dist {np.bincount(a, minlength=4)/len(a)}  switch|ok {sw.sum()/ok.sum():.3f} '
          f'agree(fixed,act|ok) {agree[0]:.3f},{agree[1]:.3f} medmaxp {np.median(p.max(1)):.3f}')
    return a

# reference: source on L2 states
a2 = tools.greedy(src, X2); ok2 = X2[:, 44] > .5; cur2 = X2[:, 40:44].argmax(1)
print('L2 source on L2 states: act dist', np.bincount(a2, minlength=4)/len(a2), 'switch|ok', ((a2 != cur2) & ok2).sum()/ok2.sum())
for nm, M in [('A avg/avg ns1', mapping()),
              ('A ns0.857', mapping(ns_scale=19.2/22.4)),
              ('B E1only/E2u', mapping(mid=(1, 0), up=(0, 1))),
              ('C avg / E2u', mapping(up=(0, 1))),
              ('D sum/sum', mapping(mid=(1, 1), up=(1, 1))),
              ('E E1only/avg', mapping(mid=(1, 0)))]:
    summarize(nm, build(M))
