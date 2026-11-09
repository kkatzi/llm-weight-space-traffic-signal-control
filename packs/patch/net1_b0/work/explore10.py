import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
SL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
q = obs[:, 0:12]*CS; d = obs[:, 12:24]*CS; qu = obs[:, 24:32]*CU; du = obs[:, 32:40]*CU
print("upstream queue (veh) quantiles per lane, by controller")
for c in range(3):
    m = ctl == c
    print(c, np.quantile(qu[m], [.5, .9, .99], axis=0).round(1))
    print("   max", qu[m].max(0).round(1), " total halted mean", (q[m].sum(1)+qu[m].sum(1)).mean().round(2))
g = tools.greedy(w, obs); P = tools.probs(w, obs)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
# states with big upstream queue on some lane
big = qu.max(1) > 4
for c in range(3):
    m = big & ok & (ctl == c)
    print("ctl", c, "big-upstream ok states", m.sum())
    for i in np.where(m)[0][:12]:
        print(f"  t={t[i]} ph={ph[i]} age={obs[i,45]*60:.0f} net={g[i]} p={P[i].round(2)} q={q[i].round(1)} qu={qu[i].round(1)}")
