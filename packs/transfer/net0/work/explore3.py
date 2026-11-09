import numpy as np
from common import *
from proxy import score
S = tools.states(); X = S['obs'].astype(float); ctrl = S['controller']
L2 = tools.states('states_L2.npz'); X2 = L2['obs'].astype(float)
src = tools.load('../source_weights_L2.npz')
print('L2 net on L2 states', score(X2, tools.greedy(src, X2), l2=True))
print('fixed ctrl recorded', score(X[ctrl == 0], S['act'][ctrl == 0]))
print('actuated recorded  ', score(X[ctrl == 1], S['act'][ctrl == 1]))
variants = {
    'A avg/avg': mapping(),
    'A ns.857': mapping(ns_scale=19.2 / 22.4),
    'B E1only/E2u': mapping(mid=(1, 0), up=(0, 1)),
    'C avg/E2u': mapping(up=(0, 1)),
    'C2 avg/.25,.75': mapping(up=(.25, .75)),
    'D sum/sum': mapping(mid=(1, 1), up=(1, 1)),
    'F E2only/E2u': mapping(mid=(0, 1), up=(0, 1)),
}
base = tools.greedy(build(variants['A avg/avg']), X)
ok = X[:, 52] > .5
for k, M in variants.items():
    a = tools.greedy(build(M), X)
    print(f'{k:16s}', {c: score(X[ctrl == c], a[ctrl == c]) for c in [0, 1]},
          'diff vs A (ok)', round(float((a[ok] != base[ok]).mean()), 3))
