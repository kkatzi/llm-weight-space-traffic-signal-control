import sys; sys.path.insert(0, '.')
import numpy as np, tools
w = tools.load("start_weights.npz"); S = tools.states()
X = S['obs'].astype(np.float64); ctrl = S['controller']
ix = {n:i for i,n in enumerate(tools.names())}
SL = {0: ['N0', 'N1', 'S0', 'S1'], 1: ['N2', 'S2'], 2: ['E0', 'E1', 'W0', 'W1'], 3: ['E2', 'W2']}
UP = {0: ['N0u', 'N1u', 'S0u', 'S1u'], 1: ['N1u', 'S1u'], 2: ['E0u', 'E1u', 'W0u', 'W1u'], 3: ['E1u', 'W1u']}
dem = np.stack([sum(X[:, ix['d_'+l]] for l in SL[p]) + sum(X[:, ix['d_'+u]] - X[:, ix['q_'+u]] for u in UP[p]) for p in range(4)], 1)
a = tools.greedy(w, X); ok = X[:,44]>0.5; ph = X[:,40:44].argmax(1)
bad = ok & (dem.sum(1)>0) & (dem[np.arange(len(X)), a]==0)
for c in range(3):
    m = bad & (ctrl==c)
    print(S['controller_names'][c], 'recorded ok-states where net picks empty phase (moving-only demand):', np.bincount(a[m], minlength=4), 'holding:', np.bincount(a[m & (a==ph)], minlength=4))
