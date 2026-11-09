"""Ablation importance of each hidden unit (replace by a constant) on recorded states, with the mirror proxy."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=200, precision=3, suppress=True)
S = tools.states(); X = S['obs'].astype(float); c = S['controller']
w = tools.load('start_weights.npz')
Y = X.copy(); Y[:, 11] = Y[:, 5]; Y[:, 23] = Y[:, 17]     # mirror view
ok = X[:, 44] > 0.5
h1, h2 = tools.hidden(w, Y)
base = (h2 @ w['W3'].T + w['b3']).argmax(1)
print('layer1: unit, mean, std, changes if set to -1, +1, mean (honoured states)')
res1 = []
for i in range(64):
    out = []
    for v in [-1.0, 1.0, h1[:, i].mean()]:
        H = h1.copy(); H[:, i] = v
        a = (np.tanh(H @ w['W2'].T + w['b2']) @ w['W3'].T + w['b3']).argmax(1)
        out.append((a[ok] != base[ok]).mean())
    res1.append((i, h1[:, i].mean(), h1[:, i].std(), *out))
for r in sorted(res1, key=lambda r: min(r[3], r[4]))[:15]:
    print('L1 %2d mean %+.2f std %.2f  ->-1 %.3f  ->+1 %.3f  ->mean %.3f' % r)
print('layer2:')
res2 = []
for j in range(64):
    out = []
    for v in [-1.0, 1.0, h2[:, j].mean()]:
        H = h2.copy(); H[:, j] = v
        a = (H @ w['W3'].T + w['b3']).argmax(1)
        out.append((a[ok] != base[ok]).mean())
    res2.append((j, h2[:, j].mean(), h2[:, j].std(), *out, *w['W3'][:, j]))
for r in sorted(res2, key=lambda r: min(r[3], r[4]))[:20]:
    print('L2 %2d mean %+.2f std %.2f  ->-1 %.3f  ->+1 %.3f  ->mean %.3f  W3 %+.2f %+.2f %+.2f %+.2f' % r)
print('b3', w['b3'])
