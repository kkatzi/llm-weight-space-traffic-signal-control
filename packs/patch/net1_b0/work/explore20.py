import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
q = obs[:, 0:12]*CS; d = obs[:, 12:24]*CS; qu = obs[:, 24:32]*CU; du = obs[:, 32:40]*CU
Ntot = d.sum(1) + du.sum(1)
Htot = q.sum(1) + qu.sum(1)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
n = len(t)
# index of step k ahead within same episode/controller
key = ctl.astype(int)*100000 + ep.astype(int)*10000
pos = {}
for i in range(n): pos[(ctl[i], ep[i], t[i])] = i
def ahead(i, k):
    return pos.get((ctl[i], ep[i], t[i] + 5*k), -1)
def outcome(i, K=12, arr=Ntot):
    s = 0.0
    for k in range(1, K+1):
        j = ahead(i, k)
        if j < 0: return np.nan
        s += (arr[j] - arr[i]) * 5
    return s
thr_ph = {0: 0, 1: 2, 2: 0, 3: 2}; left_ph = {0: 1, 1: 3, 2: 1, 3: 3}
rows = []
for a in range(4):
    blk = ok & (obs[:, 3*a+2] > 0.99) & (qu[:, 2*a+1] >= 2) & (ph == thr_ph[a]) & (q[:, 3*a+1] < 1.5) & (act != ph)
    for i in np.where(blk)[0]:
        rows.append((i, a, int(act[i] == left_ph[a])))
rows = np.array(rows)
print("blocked states with switch:", len(rows), " left chosen:", rows[:, 2].sum())
for K in [6, 12, 24]:
    oN = np.array([outcome(i, K) for i in rows[:, 0]]); oH = np.array([outcome(i, K, Htot) for i in rows[:, 0]])
    for lab, sel in [("left", rows[:, 2] == 1), ("other", rows[:, 2] == 0)]:
        print(f"K={K:2d} {lab:5s} n={sel.sum():3d}  dN veh-s={np.nanmean(oN[sel]):7.1f} ± {np.nanstd(oN[sel])/np.sqrt(sel.sum()):5.1f}   dH veh-s={np.nanmean(oH[sel]):7.1f}  start N={Ntot[rows[sel,0]].mean():.1f} H={Htot[rows[sel,0]].mean():.1f}  ctl mix={np.bincount(ctl[rows[sel,0]], minlength=3)}")
