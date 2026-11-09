import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
exec(open('work/explore24.py').read().split("g = tools.greedy")[0])
g = tools.greedy(w, obs); lg = tools.forward(w, obs)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
I = np.arange(len(t))
Ntot = d.sum(1) + du.sum(1)
pos = {(ctl[i], ep[i], t[i]): i for i in range(len(t))}
def outcome(i, K, arr):
    s = 0.0
    for k in range(1, K+1):
        j = pos.get((ctl[i], ep[i], t[i] + 5*k), -1)
        if j < 0: return np.nan
        s += (arr[j] - arr[i]) * 5
    return s
sm = Wp.copy(); sm[I, ph] = -1e9
s = sm.argmax(1)
sw = ok & (act != ph) & (t < 3400)
# arrival intensity proxy: mean N over episode window +-150s
Nwin = np.zeros(len(t))
for i in range(len(t)):
    js = [pos.get((ctl[i], ep[i], t[i] + 5*k), -1) for k in range(-30, 31)]
    js = [j for j in js if j >= 0]
    Nwin[i] = Ntot[js].mean()
sel = sw & np.isin(s, [0, 2]) & ((act == s) | np.isin(act, [1, 3]))
idx = np.where(sel)[0]
took_left = np.isin(act[idx], [1, 3]).astype(float)
for K in [12, 24]:
    y = np.array([outcome(i, K, Ntot) for i in idx]); good = ~np.isnan(y)
    for cset in [[0], [2], [0, 2], [0, 1, 2]]:
        mm = good & np.isin(ctl[idx], cset)
        ii = idx[mm]
        Xf = np.column_stack([took_left[mm], np.ones(mm.sum()), Wp[ii], obs[ii, 40:44][:, :3], Ntot[ii], Nwin[ii], q[ii], qu[ii], d[ii]] + [ (ctl[ii] == c).astype(float) for c in cset[1:]])
        beta, *_ = np.linalg.lstsq(Xf, y[mm], rcond=None)
        r = y[mm] - Xf @ beta
        cov = np.linalg.pinv(Xf.T @ Xf) * r.var()
        print(f"K={K} ctl{cset}: n={mm.sum()} left effect = {beta[0]:7.1f} ± {np.sqrt(cov[0,0]):5.1f} veh-s")
