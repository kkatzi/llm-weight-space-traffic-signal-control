import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=1, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
SL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
q = obs[:, 0:12]*CS; d = obs[:, 12:24]*CS; qu = obs[:, 24:32]*CU; du = obs[:, 32:40]*CU
Qs = np.stack([q[:, SL[p]].sum(1) for p in range(4)], 1)
g = tools.greedy(w, obs); P = tools.probs(w, obs)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
m = ok & (ctl == 0) & (g != ph)
idx = np.where(m)[0]
gap = np.array([Qs[i].max() - Qs[i, g[i]] for i in idx])
print("switch decisions:", len(idx), " chosen Qs deficit vs max quantiles:", np.quantile(gap, [.5, .75, .9, .95, .99]))
big = idx[gap >= 3]
print("deficit>=3:", len(big))
for i in big[:40]:
    print(f"ep{ep[i]} t={t[i]:4d} ph={ph[i]} -> {g[i]} P={P[i].round(2)} Qs={Qs[i]} q={q[i]} qu={qu[i]} mov_up={(du-qu)[i]}")
