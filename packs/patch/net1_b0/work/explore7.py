import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=180)
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
SL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
UP = {0: [0,1,4,5], 1: [1,5], 2: [2,3,6,7], 3: [3,7]}
q = obs[:, 0:12]*CS; d = obs[:, 12:24]*CS; qu = obs[:, 24:32]*CU; du = obs[:, 32:40]*CU
Qs = np.stack([q[:, SL[p]].sum(1) for p in range(4)], 1)
Ds = np.stack([d[:, SL[p]].sum(1) for p in range(4)], 1)
g = tools.greedy(w, obs); P = tools.probs(w, obs)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
m = ok & (ctl == 0)
idx = np.where(m)[0]
bad = [i for i in idx if Ds[i, g[i]] < .01]
print("n zero-density choices", len(bad))
for i in bad[:25]:
    print(f"t={t[i]:4d} ep={ep[i]} ph={ph[i]} age={obs[i,45]*60:.0f} choose={g[i]} p={P[i].round(2)} Qs={Qs[i]} Ds={Ds[i]} qu={qu[i].round(1)} du={du[i].round(1)}")
# time distribution of bad
print(np.histogram(t[bad], bins=[0,900,1800,2700,3600,6000]))
