import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
src = open('work/explore30.py').read().split("sel = ok")[0]
exec(src)
Htot = q.sum(1) + qu.sum(1)
tb = np.minimum(t // 600, 5)
def est(idx, a, K, arr, fe=True):
    y = np.array([outcome(i, K, arr) for i in idx]); good = ~np.isnan(y); ii = idx[good]; a = a[good]
    base = [np.ones(len(ii)), H[ii], obs[ii, 40:43], Ntot[ii], q[ii], qu[ii], d[ii], du[ii]]
    if fe:
        for e in range(6):
            for b in range(6):
                base.append(((ep[ii] == e) & (tb[ii] == b)).astype(float))
    Xf = np.column_stack([a] + base)
    beta, *_ = np.linalg.lstsq(Xf, y[good], rcond=None)
    r = y[good] - Xf @ beta; cov = np.linalg.pinv(Xf.T @ Xf) * r.var()
    return beta[0], np.sqrt(cov[0, 0]), a.mean()
def choice(ind):
    sm = ind.copy(); sm[I, ph] = -1e9; return sm.argmax(1)
for c in [0, 2]:
    sel = ok & (act != ph) & (t < 3300) & (ctl == c)
    idx = np.where(sel)[0]
    for G in [0, 10, 20]:
        s = choice(H - G*lam); a = (act[idx] == s[idx]).astype(float)
        for K in [12, 24]:
            e1 = est(idx, a, K, Ntot); e2 = est(idx, a, K, Htot)
            print(f"ctl{c} G={G:2d} K={K}: N-outcome {e1[0]:7.1f}±{e1[1]:5.1f}   halted-outcome {e2[0]:7.1f}±{e2[1]:5.1f}  (share {e1[2]:.2f})")
