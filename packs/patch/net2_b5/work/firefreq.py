import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
S = tools.states()
X = S['obs'].astype(float); C = S['controller']; T = S['time_s']
ok = X[:, 44] > 0.5
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
up0 = X[:, 32:40].sum(1) == 0
print('states with no upstream vehicles:', up0.sum(), 'of', len(X), ' times:', np.unique(T[up0] // 300 * 300, return_counts=True))
w0 = tools.load('start_weights.npz'); g0 = tools.greedy(w0, X)
for p in range(4):
    D = X[:, [12 + l for l in SERVE[p]]].sum(1)
    f = ok & (D == 0) & up0
    print('E%d fires on %d states (ok states %d); original picks p there: %d' % (p, f.sum(), ok.sum(), (g0[f] == p).sum()))
# upstream count distribution in ok states
nup = np.round(X[:, 32:40].sum(1) * 34.13)
print('ok states by #upstream vehicles (0..5):', [int(((nup == k) & ok).sum()) for k in range(6)])
