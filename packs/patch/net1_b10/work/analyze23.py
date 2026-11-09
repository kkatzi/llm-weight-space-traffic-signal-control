import numpy as np, sys
sys.path.insert(0, '.')
import tools
W0 = tools.load('start_weights.npz'); W3v = tools.load('work/v3_maxgreen.npz')
S = tools.states(); X = S['obs'].astype(np.float64); c = S['controller']
ok = X[:, 44] > 0.5
load = X[:, :12].sum(1)*3 + X[:, 24:32].sum(1)*34.1   # halting vehicles total
g0 = tools.greedy(W0, X); g3 = tools.greedy(W3v, X)
for lo, hi in [(0, 6), (6, 10), (10, 14), (14, 100)]:
    m = ok & (load >= lo) & (load < hi)
    print(f'halting in [{lo},{hi}): n={m.sum()} v3 change rate {(g0[m]!=g3[m]).mean():.4f}')
# per-unit ablation impact by load
W1, b1, W2, b2, W3, b3 = (W0[k] for k in ['W1','b1','W2','b2','W3','b3'])
h1 = np.tanh(X @ W1.T + b1); h2 = np.tanh(h1 @ W2.T + b2); g = (h2 @ W3.T + b3).argmax(1)
hi_m = ok & (load >= 10)
res = []
for j in range(64):
    hh = h1.copy(); hh[:, j] = h1[:, j].mean()
    g2 = (np.tanh(hh @ W2.T + b2) @ W3.T + b3).argmax(1)
    res.append(((g2[ok] != g[ok]).mean(), (g2[hi_m] != g[hi_m]).mean(), j))
res.sort(key=lambda r: r[0] + r[1])
print('h1 units by ablation (all ok, high-load ok):')
for r in res[:8]: print('   unit %2d  all %.4f  high %.4f' % (r[2], r[0], r[1]))
