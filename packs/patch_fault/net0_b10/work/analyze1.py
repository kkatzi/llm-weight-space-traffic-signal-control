import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w0 = tools.load('start_weights.npz')
S = tools.states(); X = S['obs'].astype(np.float64); C = S['controller']
ok = X[:, 44] > 0.5

def proxy(w, a=1.0):
    w = {k: v.copy() for k, v in w.items()}
    w['W1'][:, 5] += a * w['W1'][:, 11]
    w['W1'][:, 17] += a * w['W1'][:, 23]
    return w

g0 = tools.greedy(w0, X)
for a in [0.5, 1.0]:
    wp = proxy(w0, a)
    gp = tools.greedy(wp, X)
    print('a', a, 'edit', tools.edit_size(wp, w0))
    print(' agreement all', (g0 == gp).mean(), 'honoured', (g0[ok] == gp[ok]).mean())
    for c in range(3):
        m = (C == c) & ok
        print('  ctrl', c, 'phase dist failed', np.bincount(g0[m], minlength=4) / m.sum(), 'proxy', np.bincount(gp[m], minlength=4) / m.sum())
    # transitions confusion on honoured
    M = np.zeros((4, 4), int)
    for i, j in zip(g0[ok], gp[ok]): M[i, j] += 1
    print(M)

# sensitivity: set x11, x23 to values and see decision distribution
for q in [0, 1/3, 2/3, 1]:
    Xs = X.copy(); Xs[:, 11] = q; Xs[:, 23] = q
    g = tools.greedy(w0, Xs)
    print('W2 set to', round(q, 2), 'phase dist (ok)', np.bincount(g[ok], minlength=4) / ok.sum())
for q in [0, 1/3, 2/3, 1]:
    Xs = X.copy(); Xs[:, 5] = q; Xs[:, 17] = q
    g = tools.greedy(w0, Xs)
    print('E2 set to', round(q, 2), 'phase dist (ok)', np.bincount(g[ok], minlength=4) / ok.sum())
