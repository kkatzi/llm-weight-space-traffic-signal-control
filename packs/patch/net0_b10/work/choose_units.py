"""Pick which hidden units to sacrifice: count decisions (ok-states, all controllers) that change for reasons
other than the empty-phase rule, for single units (sacrificed = held at constant), then combos of the best."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
w0 = tools.load('start_weights.npz')
X = tools.states()['obs'].astype(np.float64)
ok = X[:, 44] > 0.5
Xo = X[ok]
h1, h2 = tools.hidden(w0, Xo)
a0 = tools.greedy(w0, Xo)
def decide(k1s, k2s):
    H1 = h1.copy()
    for k, c in k1s: H1[:, k] = c
    H2 = np.tanh(H1 @ w0['W2'].T + w0['b2'])
    b3 = w0['b3'].copy()
    for j in k2s:
        b3 += w0['W3'][:, j] * h2[:, j].mean()
        H2[:, j] = 0.0
    return (H2 @ w0['W3'].T + b3).argmax(1)
c1 = {}
for k in range(64):
    best = min(((decide([(k, c)], []) != a0).sum(), c) for c in (-1.0, 1.0))
    c1[k] = best
r1 = sorted(range(64), key=lambda k: c1[k][0])[:10]
print('h1 singles:', [(k, int(c1[k][0]), c1[k][1]) for k in r1])
d2 = {j: (decide([], [j]) != a0).sum() for j in range(64)}
r2 = sorted(range(64), key=lambda j: d2[j])[:10]
print('h2 singles:', [(j, int(d2[j])) for j in r2])
best = None
for ks in itertools.combinations(r1, 4):
    k1s = [(k, c1[k][1]) for k in ks]
    a = decide(k1s, [])
    n = (a != a0).sum()
    if best is None or n < best[0]: best = (n, ks)
print('best h1 4-combo', best)
k1s = [(k, c1[k][1]) for k in best[1]]
best2 = None
for js in itertools.combinations(r2, 4):
    n = (decide(k1s, list(js)) != a0).sum()
    if best2 is None or n < best2[0]: best2 = (n, js)
print('best h2 4-combo with that h1 combo', best2)
print('current choice:', (decide([(32,1.),(60,1.),(57,1.),(59,-1.)], [6,37,49,0]) != a0).sum())
