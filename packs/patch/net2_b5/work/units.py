import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=3, suppress=True)
w = tools.load('start_weights.npz')
S = tools.states()
X = S['obs'].astype(float)
ok = X[:, 44] > 0.5
h1, h2 = tools.hidden(w, X)
g0 = tools.greedy(w, X)
def fwd_h1(H1):
    H2 = np.tanh(H1 @ w['W2'].T + w['b2']); return (H2 @ w['W3'].T + w['b3']).argmax(1)
def fwd_h2(H2):
    return (H2 @ w['W3'].T + w['b3']).argmax(1)
res1 = []
for j in range(64):
    H = h1.copy(); H[:, j] = h1[:, j].mean()
    res1.append(((fwd_h1(H) != g0)[ok].mean(), j, h1[:, j].mean(), h1[:, j].std(), np.abs(w['W2'][:, j]).sum()))
res1.sort()
print('h1 units: (flip rate on ok states, unit, mean, std, |W2 col| sum)')
for r in res1[:12]: print('  %.4f %2d mean %.3f std %.3f colL1 %.2f' % r)
res2 = []
for i in range(64):
    H = h2.copy(); H[:, i] = h2[:, i].mean()
    res2.append(((fwd_h2(H) != g0)[ok].mean(), i, h2[:, i].mean(), h2[:, i].std(), (np.abs(h2[:, i]) > 0.99).mean(), w['W3'][:, i]))
res2.sort(key=lambda r: r[0])
print('h2 units: (flip rate, unit, mean, std, frac |h|>0.99, W3 col)')
for r in res2[:14]: print('  %.4f %2d mean %.3f std %.3f sat %.2f' % r[:5], r[5].round(2))
