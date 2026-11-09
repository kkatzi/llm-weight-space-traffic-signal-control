import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
exec(open('work/explore24.py').read().split("g = tools.greedy")[0])
lg = tools.forward(w, obs); g = lg.argmax(1)
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
Nwin = np.zeros(len(t))
for i in range(len(t)):
    js = [pos.get((ctl[i], ep[i], t[i] + 5*k), -1) for k in range(-30, 31)]
    js = [j for j in js if j >= 0]; Nwin[i] = Ntot[js].mean()
H = W(obs, near=0.0)
sel = ok & (act != ph) & (t < 3400) & (ctl != 1)   # fixed-time excluded: its actions don't depend on state
idx = np.where(sel)[0]
for G in [0, 10, 20]:
    ind = H - G*lam; sm = ind.copy(); sm[I, ph] = -1e9; s = sm.argmax(1)
    for K in [12, 24]:
        y = np.array([outcome(i, K, Ntot) for i in idx]); good = ~np.isnan(y); ii = idx[good]
        a_idx = (act[ii] == s[ii]).astype(float); a_net = (act[ii] == g[ii]).astype(float)
        base = [np.ones(len(ii)), H[ii], obs[ii, 40:43], Ntot[ii], Nwin[ii], q[ii], qu[ii], d[ii], (ctl[ii] == 2).astype(float)]
        for nm, a in [("follow-index", a_idx), ("follow-net", a_net)]:
            Xf = np.column_stack([a] + base)
            beta, *_ = np.linalg.lstsq(Xf, y[good], rcond=None)
            r = y[good] - Xf @ beta; cov = np.linalg.pinv(Xf.T @ Xf) * r.var()
            print(f"G={G} K={K} {nm:13s} share={a.mean():.2f} effect={beta[0]:7.1f} ± {np.sqrt(cov[0,0]):5.1f} veh-s")
