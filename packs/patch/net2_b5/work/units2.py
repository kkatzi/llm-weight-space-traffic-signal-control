import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
w = tools.load('start_weights.npz')
X = tools.states()['obs'].astype(float)
ok = X[:, 44] > 0.5
h1, h2 = tools.hidden(w, X)
g0 = tools.greedy(w, X)
def dec(H1):
    H2 = np.tanh(H1 @ w['W2'].T + w['b2']); return (H2 @ w['W3'].T + w['b3']).argmax(1)
rows = []
for j in range(64):
    r = [j, h1[:, j].mean()]
    for v in [-1.0, 1.0, h1[:, j].mean()]:
        H = h1.copy(); H[:, j] = v
        r.append((dec(H) != g0)[ok].mean())
    rows.append(r)
rows.sort(key=lambda r: min(r[2], r[3]))
print('unit, mean, flip(-1), flip(+1), flip(mean)')
for r in rows[:15]:
    print('%2d %.3f  %.4f %.4f %.4f' % tuple(r))
