import sys; sys.path.insert(0, '.')
import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states()
for k,v in S.items(): print(k, v.shape, v.dtype, v[:5] if v.ndim==1 else '')
print(S.get('controller_names'))
X = S['obs']; ctrl = S['controller']; ep = S['episode']; t = S['time_s']; act = S['act']
for c in np.unique(ctrl):
    m = ctrl==c
    print(c, m.sum(), 'episodes', np.unique(ep[m]))
g = tools.greedy(w, X)
m0 = ctrl==0
print('agreement net vs recorded act (start_network):', (g[m0]==act[m0]).mean())
print(tools.trainability(w))
for k in tools.KEYS: print(k, w[k].shape, np.abs(w[k]).max(), np.abs(w[k]).mean())
