import sys; sys.path.insert(0, '.')
import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states()
print({k: (v.shape, v.dtype) for k, v in S.items()})
print(S['controller_names'] if 'controller_names' in S else None)
obs, act, ep, t, ctl = S['obs'], S['act'], S['episode'], S['time_s'], S['controller']
for c in np.unique(ctl):
    m = ctl == c
    print("ctl", c, m.sum(), "episodes", np.unique(ep[m]))
g = tools.greedy(w, obs)
m = ctl == 0
print("agreement net greedy vs recorded act for start_network:", (g[m] == act[m]).mean())
print(tools.trainability(w))
for k in tools.KEYS: print(k, w[k].shape, np.abs(w[k]).max())
