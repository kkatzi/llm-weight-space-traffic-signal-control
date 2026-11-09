import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(linewidth=220, precision=3, suppress=True)
w = tools.load("start_weights.npz")
S = tools.states(); X = S['obs'].astype(float); a = tools.greedy(w, X)
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
Xo, ao, po = X[ok], a[ok], ph[ok]
L = {n: i for i, n in enumerate(['N0','N1','N2','E0','E1','E2','S0','S1','S2','W0','W1','W2'])}
def v(X, n): return 3*(X[:, L[n]] + X[:, 12 + L[n]])  # halting counts twice, moving once (vehicles)
def scores(X, w0):
    return np.stack([v(X,'N1') + v(X,'S1') + w0*(v(X,'N0') + v(X,'S0')),
                     v(X,'N2') + v(X,'S2'),
                     v(X,'E1') + v(X,'W1') + w0*(v(X,'E0') + v(X,'W0')),
                     v(X,'E2') + v(X,'W2')], 1)
for w0 in [0, 0.25, 0.5, 0.75, 1.0]:
    P = scores(Xo, w0); P[np.arange(len(P)), po] = -99
    r = P.argmax(1)
    print("w0", w0, "agree", (r == ao).mean().round(4), [(r[po==p] == ao[po==p]).mean().round(3) for p in range(4)])
