import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=3, suppress=True)
w = tools.load('start_weights.npz')
S = tools.states()
X = S['obs'].astype(float)
h1, h2 = tools.hidden(w, X)
J, I = 50, 1
pre_wo = h1 @ w['W2'][I].T - h1[:, J] * w['W2'][I, J] + w['b2'][I]
print('unit1 pre without h1_50: min %.2f max %.2f' % (pre_wo.min(), pre_wo.max()))
print('W3 col 1', w['W3'][:, I], 'b3', w['b3'])
print('h1_50 mean', h1[:, J].mean(), 'h2_1 mean', h2[:, I].mean())
print('W2 col 50 abs max', np.abs(w['W2'][:, J]).max())
# also what the network would pick among 0..2 in the bad states
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
g = tools.greedy(w, X); lg = tools.forward(w, X)
bad = ok & (ph == 3) & (g == 3) & (X[:, 17] + X[:, 23] == 0)
alt = lg[bad][:, :3].argmax(1)
print('bad', bad.sum(), 'alt picks', np.bincount(alt, minlength=3))
