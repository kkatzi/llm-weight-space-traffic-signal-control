"""Offline analysis: how decisions change when the missing W2 inputs are imputed in different ways."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools

np.set_printoptions(precision=3, suppress=True, linewidth=200)
S = tools.states(); X = S['obs'].astype(float); c = S['controller']
w = tools.load('start_weights.npz')
ok = X[:, 44] > 0.5

def imp(X, fq, fd):
    Y = X.copy(); Y[:, 11] = fq(X); Y[:, 23] = fd(X); return Y

cands = {
    'zero (as deployed)': (lambda X: 0 * X[:, 0], lambda X: 0 * X[:, 0]),
    'mirror E2': (lambda X: X[:, 5], lambda X: X[:, 17]),
    'mirror N2/S2 avg': (lambda X: (X[:, 2] + X[:, 8]) / 2, lambda X: (X[:, 14] + X[:, 20]) / 2),
    'const mean E2': (lambda X: 0 * X[:, 0] + X[c == 1, 5].mean(), lambda X: 0 * X[:, 0] + X[c == 1, 17].mean()),
    'full': (lambda X: 0 * X[:, 0] + 1, lambda X: 0 * X[:, 0] + 1),
}
base = tools.greedy(w, X)
for name, (fq, fd) in cands.items():
    Y = imp(X, fq, fd)
    a = tools.greedy(w, Y)
    print(f'{name:22s}', 'changed(ok)', round(float((a != base)[ok].mean()), 3),
          'dist(ok)', np.bincount(a[ok], minlength=4) / ok.sum())
