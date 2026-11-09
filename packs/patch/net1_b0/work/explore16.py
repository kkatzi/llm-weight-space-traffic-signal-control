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
thr_ph = {0: 0, 1: 2, 2: 0, 3: 2}; left_ph = {0: 1, 1: 3, 2: 1, 3: 3}
for c in [0]:
    for a, nm in enumerate("NESW"):
        pocket_full = obs[:, 3*a+2] > 0.99
        blk = ok & (ctl == c) & pocket_full & (qu[:, 2*a+1] >= 2) & (ph == thr_ph[a]) & (q[:, 3*a+1] < 1.5)
        for i in np.where(blk & (g != left_ph[a]))[0]:
            print(f"{nm} ep{ep[i]} t={t[i]} ph={ph[i]} net={g[i]} P={P[i].round(2)} Qs={Qs[i]} q={q[i]} qu={qu[i]} du={du[i]}")
