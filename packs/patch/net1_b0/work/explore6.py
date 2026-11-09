import sys; sys.path.insert(0, '.')
import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
q = obs[:, 0:12]*CS; d = obs[:, 12:24]*CS; qu = obs[:, 24:32]*CU; du = obs[:, 32:40]*CU
# phase lanes: stop-line indices
SL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
UP = {0: [0,1,4,5], 1: [1,5], 2: [2,3,6,7], 3: [3,7]}
def phase_stats(o):
    q = o[:, 0:12]*CS; d = o[:, 12:24]*CS; qu = o[:, 24:32]*CU; du = o[:, 32:40]*CU
    Qs = np.stack([q[:, SL[p]].sum(1) for p in range(4)], 1)
    Ds = np.stack([d[:, SL[p]].sum(1) for p in range(4)], 1)
    Qu = np.stack([qu[:, UP[p]].sum(1) for p in range(4)], 1)
    return Qs, Ds, Qu
Qs, Ds, Qu = phase_stats(obs)
g = tools.greedy(w, obs)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
m = ok & (ctl == 0)
print("network decisions with ok:", m.sum())
ch = g[m]
idx = np.where(m)[0]
qs_ch = Qs[idx, ch]; ds_ch = Ds[idx, ch]
others_max = np.array([np.max(np.delete(Qs[i], c)) for i, c in zip(idx, ch)])
print("chosen phase stopline density 0 share:", (ds_ch < 0.01).mean())
print("chosen phase stopline queue 0 share:", (qs_ch < 0.01).mean())
print("chosen has zero density but some other has queue>=2:", ((ds_ch < .01) & (others_max >= 2)).mean())
# compare with argmax of stopline queue
amax = Qs[idx].argmax(1)
print("agreement with max stopline queue phase:", (amax == ch).mean())
# hold decisions
hold = ch == ph[idx]
print("holds:", hold.sum())
# distribution of chosen phase queue rank
rank = np.array([(Qs[i] > Qs[i, c]).sum() for i, c in zip(idx, ch)])
print("rank of chosen phase by stopline queue (0=max):", np.bincount(rank)/len(rank))
rankD = np.array([(Ds[i] + Qu[i] > Ds[i, c] + Qu[i, c]).sum() for i, c in zip(idx, ch)])
print("rank by density+upqueue:", np.bincount(rankD)/len(rankD))
# all controllers: same for whole set
for c in range(3):
    mm = ok & (ctl == c); ii = np.where(mm)[0]; cc = g[mm]
    print("ctl", c, "net choice zero-density share", (Ds[ii, cc] < .01).mean(), " current phase Ds mean", Ds[ii, ph[ii]].mean())
