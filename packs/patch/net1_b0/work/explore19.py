import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
SL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
q = obs[:, 0:12]*CS; d = obs[:, 12:24]*CS; qu = obs[:, 24:32]*CU; du = obs[:, 32:40]*CU
Qs = np.stack([q[:, SL[p]].sum(1) for p in range(4)], 1)
lg = tools.forward(w, obs); P = tools.probs(w, obs); g = lg.argmax(1)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
srt = np.sort(lg, 1); gap = srt[:, -1] - srt[:, -2]
second = np.argsort(lg, 1)[:, -2]
m = ok & (ctl == 0)
print("honoured decisions logit gap quantiles:", np.quantile(gap[m], [.05, .1, .25, .5]))
for thr in [0.25, 0.5, 1.0]:
    mm = m & (gap < thr)
    print(f"gap<{thr}: n={mm.sum()} ({mm.sum()/m.sum():.3f}); pairs:", np.unique(np.stack([g[mm], second[mm]],1), axis=0, return_counts=True)[1].sum())
    # in these, which has more halted?
    I = np.where(mm)[0]
    more = (Qs[I, second[I]] > Qs[I, g[I]]).mean(); less = (Qs[I, second[I]] < Qs[I, g[I]]).mean()
    print(f"   runner-up has more halted: {more:.2f}, fewer: {less:.2f}")
from collections import Counter
mm = m & (gap < 1.0)
print(Counter([(ph[i], g[i], second[i]) for i in np.where(mm)[0]]).most_common(20))
