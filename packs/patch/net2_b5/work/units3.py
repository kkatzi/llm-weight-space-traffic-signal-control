import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
w = tools.load('start_weights.npz')
X = tools.states()['obs'].astype(float)
ok = X[:, 44] > 0.5
h1, h2 = tools.hidden(w, X)
g0 = tools.greedy(w, X)
def dec(H1, b2):
    H2 = np.tanh(H1 @ w['W2'].T + b2); return (H2 @ w['W3'].T + w['b3']).argmax(1)
for j in [42, 38, 5, 51, 6, 48, 43, 49, 19, 11, 0, 4, 62, 58]:
    m = h1[:, j].mean()
    out = []
    for v in [-1.0, 1.0]:
        shift = w['W2'][:, j] * (v - m)          # pre-activation shift if pinned at v
        order = np.argsort(-np.abs(shift))
        res = []
        for ncomp in [0, 10, 20, 30, 63]:
            b2 = w['b2'].copy()
            b2[order[:ncomp]] -= shift[order[:ncomp]]
            H = h1.copy(); H[:, j] = v
            res.append(round(float((dec(H, b2) != g0)[ok].mean()) * 100, 2))
        out.append((v, res))
    print(j, 'mean %.2f' % m, out)
