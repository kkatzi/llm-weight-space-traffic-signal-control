import sys; sys.path.insert(0, '.')
import numpy as np, tools
w = tools.load("start_weights.npz"); S = tools.states()
X = S['obs'].astype(np.float64); ctrl = S['controller']
ix = {n:i for i,n in enumerate(tools.names())}
cs, cu = 22.4/7.5, 256/7.5
SL = {0: ['N0', 'N1', 'S0', 'S1'], 1: ['N2', 'S2'], 2: ['E0', 'E1', 'W0', 'W1'], 3: ['E2', 'W2']}
UP = {0: ['N0u', 'N1u', 'S0u', 'S1u'], 1: ['N1u', 'S1u'], 2: ['E0u', 'E1u', 'W0u', 'W1u'], 3: ['E1u', 'W1u']}
veh = np.stack([sum(X[:, ix['d_'+l]] for l in SL[p])*cs + sum(X[:, ix['d_'+u]] for u in UP[p])*cu for p in range(4)], 1)
qsl = np.stack([sum(X[:, ix['q_'+l]] for l in SL[p])*cs for p in range(4)], 1)
a = tools.greedy(w, X); ok = X[:,44]>0.5; ph = X[:,40:44].argmax(1)
m = (ctrl==0) & ok & (a!=ph)
ch = veh[np.arange(len(X)), a]; chq = qsl[np.arange(len(X)), a]
oth = qsl.copy(); oth[np.arange(len(X)), a] = -1; oth[np.arange(len(X)), ph] = -1
mo = oth.max(1)
for p in range(4):
    mm = m & (a==p)
    print('switch to phase', p, 'n', mm.sum(), ' chosen-phase halted at stop line: mean %.2f' % chq[mm].mean(),
          ' cases chosen halted <=1 while another non-current phase has >=3 halted: %d' % ((chq[mm] <= 1.01) & (mo[mm] >= 2.99)).sum())
