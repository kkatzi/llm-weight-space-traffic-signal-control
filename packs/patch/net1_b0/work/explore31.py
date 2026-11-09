import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
src = open('work/explore30.py').read().split("sel = ok")[0]
exec(src)
lgap = np.sort(lg, 1); gapv = lgap[:, -1] - lgap[:, -2]
for cset in [[0], [2]]:
    sel = ok & (act != ph) & (t < 3400) & np.isin(ctl, cset)
    idx = np.where(sel)[0]
    for G in [0, 10]:
        ind = H - G*lam; sm = ind.copy(); sm[I, ph] = -1e9; s = sm.argmax(1)
        for K in [6, 12, 24]:
            y = np.array([outcome(i, K, Ntot) for i in idx]); good = ~np.isnan(y); ii = idx[good]
            a_idx = (act[ii] == s[ii]).astype(float)
            base = [np.ones(len(ii)), H[ii], obs[ii, 40:43], Ntot[ii], Nwin[ii], q[ii], qu[ii], d[ii], du[ii]]
            Xf = np.column_stack([a_idx] + base)
            beta, *_ = np.linalg.lstsq(Xf, y[good], rcond=None)
            r = y[good] - Xf @ beta; cov = np.linalg.pinv(Xf.T @ Xf) * r.var()
            print(f"ctl{cset} G={G} K={K} follow-index share={a_idx.mean():.2f} effect={beta[0]:7.1f} ± {np.sqrt(cov[0,0]):5.1f}")
