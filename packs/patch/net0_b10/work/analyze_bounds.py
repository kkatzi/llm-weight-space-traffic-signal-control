import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states(); X = S['obs'].astype(np.float64)
h1, h2 = tools.hidden(w, X)
a1 = X @ w['W1'].T + w['b1']; a2 = h1 @ w['W2'].T + w['b2']
bnd2 = np.abs(w['W2']).sum(1) + np.abs(w['b2'])
print('h2 pre-act worst-case bound (|W2 row|_1+|b2|):'); print(bnd2)
print('observed |a2| max:'); print(np.abs(a2).max(0))
print('W1 row L1 + |b1|:', (np.abs(w['W1']).sum(1)+np.abs(w['b1'])).round(1))
print('logit range', tools.forward(w,X).min(0), tools.forward(w,X).max(0))
print('logit gap top1-top4 max', np.ptp(tools.forward(w,X),axis=1).max())
