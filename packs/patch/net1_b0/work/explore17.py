import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
SL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
UPo = {0: [0,1,4,5], 1: [1,5], 2: [2,3,6,7], 3: [3,7]}
q = obs[:, 0:12]*CS; d = obs[:, 12:24]*CS; qu = obs[:, 24:32]*CU; du = obs[:, 32:40]*CU
Qs = np.stack([q[:, SL[p]].sum(1) for p in range(4)], 1)
Qu = np.stack([qu[:, UPo[p]].sum(1) for p in range(4)], 1)
g = tools.greedy(w, obs)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
I = np.arange(len(t))
m0 = ok & (ctl == 0)
for e in range(6):
    m = m0 & (ep == e)
    for b0, b1 in [(0, 900), (900, 1800), (1800, 2700), (2700, 3600)]:
        mm = m & (t >= b0) & (t < b1)
        resid = Qs[I, ph][mm]
        print(f"ep{e} {b0:4d}-{b1:4d}: n={mm.sum():3d} cur residual stopline halted mean={resid.mean():.2f} share>0={(resid>0.5).mean():.2f}  share>=2={(resid>=1.5).mean():.2f}  others Qs mean={(Qs[mm].sum(1)-resid).mean():.1f}  holds={(g[mm]==ph[mm]).mean():.2f}")
