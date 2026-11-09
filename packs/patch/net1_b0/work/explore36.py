import sys; sys.path.insert(0, '.')
import numpy as np, tools
src = open('work/explore33.py').read().split("for c in [0, 2]:")[0]
exec(src)
lam = np.array([0.175, 0.075, 0.2917, 0.125])
G = 10
for alpha in [0.5, 1.0, 2.0]:
    l2 = lg + alpha*(H - G*lam); g2 = l2.argmax(1)
    for c in [1, 2]:
        sel = ok & (t < 3300) & (ctl == c) & (g2 != g) & ((act == g) | (act == g2))
        idx = np.where(sel)[0]
        a = (act[idx] == g2[idx]).astype(float)
        if len(idx) < 20: print(c, alpha, "too few", len(idx)); continue
        # simple regression without FE (small n)
        for K, arr, nm in [(12, Htot, "H12"), (24, Htot, "H24"), (24, Ntot, "N24")]:
            y = np.array([outcome(i, K, arr) for i in idx]); good = ~np.isnan(y); ii = idx[good]; aa = a[good]
            Xf = np.column_stack([aa, np.ones(len(ii)), H[ii], Ntot[ii], obs[ii, 40:43]])
            beta, *_ = np.linalg.lstsq(Xf, y[good], rcond=None)
            r = y[good] - Xf @ beta; cov = np.linalg.pinv(Xf.T @ Xf) * r.var()
            print(f"alpha={alpha} ctl{c} n={len(ii)} took-new share={aa.mean():.2f} {nm}: effect of new choice {beta[0]:7.1f} ± {np.sqrt(cov[0,0]):5.1f}")
