import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
src = open('work/explore30.py').read().split("sel = ok")[0]
exec(src)
rng = np.random.default_rng(0)
def est(idx, a, K):
    y = np.array([outcome(i, K, Ntot) for i in idx]); good = ~np.isnan(y); ii = idx[good]; a = a[good]
    base = [np.ones(len(ii)), H[ii], obs[ii, 40:43], Ntot[ii], Nwin[ii], q[ii], qu[ii], d[ii], du[ii]]
    Xf = np.column_stack([a] + base)
    beta, *_ = np.linalg.lstsq(Xf, y[good], rcond=None)
    r = y[good] - Xf @ beta; cov = np.linalg.pinv(Xf.T @ Xf) * r.var()
    return beta[0], np.sqrt(cov[0, 0]), a.mean()
sel = ok & (act != ph) & (t < 3300) & (ctl == 0)
idx = np.where(sel)[0]
def choice(ind):
    sm = ind.copy(); sm[I, ph] = -1e9; return sm.argmax(1)
for nm, ind in [("index G=10", H - 10*lam), ("index G=0", H), ("anti: min H", -H + 0*lam), ("left-pref", H + np.array([0, 3, 0, 3])), ("thru-pref", H + np.array([3, 0, 3, 0]))]:
    s = choice(ind)
    for K in [12, 24, 36]:
        e, se, sh = est(idx, (act[idx] == s[idx]).astype(float), K)
        print(f"{nm:12s} K={K}: share={sh:.2f} effect={e:7.1f} ± {se:5.1f}")
# near-tie subset
lgs = np.sort(lg, 1); gapv = lgs[:, -1] - lgs[:, -2]
sel2 = sel & (gapv < 1.5)
idx2 = np.where(sel2)[0]
s = choice(H - 10*lam)
for K in [12, 24, 36]:
    e, se, sh = est(idx2, (act[idx2] == s[idx2]).astype(float), K)
    print(f"near-tie (n={len(idx2)}) index G=10 K={K}: share={sh:.2f} effect={e:7.1f} ± {se:5.1f}")
