import sys; sys.path.insert(0, '.')
import numpy as np, tools
from collections import Counter
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
SL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
q = obs[:, 0:12]*CS; d = obs[:, 12:24]*CS; qu = obs[:, 24:32]*CU; du = obs[:, 32:40]*CU
Qs = np.stack([q[:, SL[p]].sum(1) for p in range(4)], 1)
g = tools.greedy(w, obs); P = tools.probs(w, obs)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
m = ok & (ctl == 0)
idx = np.where(m)[0]
# transitions chosen at ok, with current phase
tr = Counter((ph[i], g[i]) for i in idx)
print(sorted(tr.items()))
# for each (cur, choice), mean Qs vector and mean upstream queues
for (c, ch), n in sorted(tr.items()):
    ii = [i for i in idx if ph[i] == c and g[i] == ch]
    print(f"cur{c}->{ch} n={n:3d} Qs={Qs[ii].mean(0)} qu={qu[ii].mean(0).round(1)} q_pockets(N2,E2,S2,W2)={q[ii][:, [2,5,8,11]].mean(0)}")
