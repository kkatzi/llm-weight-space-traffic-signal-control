import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
for k,v in S.items(): print(k, v.shape, v.dtype, v[:3] if v.ndim==1 else '')
print(S['controller_names'] if 'controller_names' in S else '')
X = S['obs']; ctrl = S['controller']; ep = S['episode']; act = S['act']; t = S['time_s']
a = tools.greedy(w, X)
m = ctrl==0
print("net agreement on own states:", (a[m]==act[m]).mean())
print(tools.trainability(w))
# phase distribution
for c in range(3):
    mm = ctrl==c
    print(c, np.bincount(a[mm], minlength=4), np.bincount(act[mm], minlength=4))
# episodes for net
for e in np.unique(ep[m]):
    mm = m & (ep==e)
    print('ep', e, mm.sum(), t[mm].min(), t[mm].max())
