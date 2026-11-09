import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); c = S['controller']
ok = X[:, 44] > 0.5
W1, b1, W2, b2, W3, b3 = (w[k] for k in ['W1','b1','W2','b2','W3','b3'])
h1 = np.tanh(X @ W1.T + b1); h2 = np.tanh(h1 @ W2.T + b2)
g = (h2 @ W3.T + b3).argmax(1)
m = ok
res1 = []
for j in range(64):
    hh = h1.copy(); hh[:, j] = h1[:, j].mean()
    g2 = (np.tanh(hh @ W2.T + b2) @ W3.T + b3).argmax(1)
    res1.append(((g2[m] != g[m]).mean(), j, h1[:, j].std(), np.abs(W2[:, j]).sum()))
res1.sort()
print('h1 units by mean-ablation decision change (ok states): rate, unit, std, sum|W2[:,j]|')
for r in res1[:10]: print('   %.4f  %2d  %.3f  %.2f' % r)
res2 = []
for k in range(64):
    hh = h2.copy(); hh[:, k] = h2[:, k].mean()
    g2 = (hh @ W3.T + b3).argmax(1)
    res2.append(((g2[m] != g[m]).mean(), k, h2[:, k].std(), np.abs(W3[:, k]).sum()))
res2.sort()
print('h2 units by mean-ablation decision change (ok states): rate, unit, std, sum|W3[:,k]|')
for r in res2[:10]: print('   %.4f  %2d  %.3f  %.2f' % r)
