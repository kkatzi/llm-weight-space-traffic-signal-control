import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(linewidth=220, precision=3, suppress=True)
w = tools.load("start_weights.npz")
S = tools.states(); X = S['obs'].astype(float); a = tools.greedy(w, X)
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
Xo, ao, po = X[ok], a[ok], ph[ok]
def scores(X, kind):
    q = X[:, 0:12]; d = X[:, 12:24]
    v = {'q': q, 'd': d, 'qd': q + d}[kind]
    P = np.stack([(v[:, 0] + v[:, 1] + v[:, 6] + v[:, 7])/2, v[:, 2] + v[:, 8],
                  (v[:, 3] + v[:, 4] + v[:, 9] + v[:, 10])/2, v[:, 5] + v[:, 11]], 1)
    return P
for kind in ['q', 'd', 'qd']:
    P = scores(Xo, kind)
    P2 = P.copy(); P2[np.arange(len(P2)), po] = -9
    r = P2.argmax(1)
    print(kind, "per-lane argmax excl current:", (r == ao).mean())
    for p in range(4):
        m = po == p
        M = np.zeros((4, 4), int)
        for i, j in zip(ao[m], r[m]): M[i, j] += 1
        print(" phase", p, "agree", (r[m] == ao[m]).mean().round(3)); print(M)
