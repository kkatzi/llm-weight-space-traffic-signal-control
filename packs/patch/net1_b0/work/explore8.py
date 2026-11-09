import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=180)
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
SL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
UP = {0: [0,4], 1: [], 2: [2,6], 3: []}   # exclusively-owned upstream lanes
UPS = {0: [1,5], 1: [1,5], 2: [3,7], 3: [3,7]}  # shared upstream lanes
q = obs[:, 0:12]*CS; d = obs[:, 12:24]*CS; qu = obs[:, 24:32]*CU; du = obs[:, 32:40]*CU
Qs = np.stack([q[:, SL[p]].sum(1) for p in range(4)], 1)
Qtot = np.stack([q[:, SL[p]].sum(1) + qu[:, UP[p]].sum(1) + 0.5*qu[:, UPS[p]].sum(1) for p in range(4)], 1)
Mov = np.stack([(d-q)[:, SL[p]].sum(1) + (du-qu)[:, UP[p]].sum(1)*0.2 + (du-qu)[:, UPS[p]].sum(1)*0.1 for p in range(4)], 1)
g = tools.greedy(w, obs)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
I = np.arange(len(t))
cur_mov = Mov[I, ph]
others_q = Qtot.sum(1) - Qtot[I, ph]
for c in range(3):
    m = ok & (ctl == c)
    print("ctl", c)
    for thr in [0.5, 1, 2, 3]:
        mm = m & (others_q <= thr)
        print(f"  others halted <= {thr}: share {mm.sum()/m.sum():.3f}; net holds {(g[mm]==ph[mm]).mean():.2f}; mean cur moving {cur_mov[mm].mean():.2f}; cur moving>1 share {(cur_mov[mm]>1).mean():.2f}")
# network's own: time profile of others_q at switches
m = ok & (ctl == 0)
print("network ok-decisions others_q quantiles", np.quantile(others_q[m], [.1,.25,.5,.75,.9]))
print("network ok-decisions cur Qtot quantiles", np.quantile(Qtot[I, ph][m], [.1,.25,.5,.75,.9]))
print("network ok-decisions cur moving quantiles", np.quantile(cur_mov[m], [.1,.25,.5,.75,.9]))
