import sys; sys.path.insert(0, '.')
import numpy as np, tools
src = open('work/explore33.py').read().split("for c in [0, 2]:")[0]
exec(src)
sys.path.insert(0, 'work')
from build import build, H_coef
Hh = obs @ H_coef().T
w1 = build(w, alpha=0.75, M=2.5, G=5, X_mean=obs[ok], k_units=(31, 27, 59), j_units=(25, 35, 41))
g1 = tools.greedy(w1, obs)
others = Hh.sum(1) - Hh[I, ph]
for c in [0, 2]:
    sel = ok & (t < 3300) & (ctl == c)
    idx = np.where(sel)[0]
    a = (act[idx] == ph[idx]).astype(float)
    for K, arr, nm in [(12, Htot, "H12"), (24, Htot, "H24"), (24, Ntot, "N24")]:
        e = est(idx, a, K, arr)
        print(f"ctl{c} all ok states: hold share {a.mean():.3f}  {nm} effect of hold {e[0]:7.1f} ± {e[1]:5.1f}")
    # restrict to states where the new network would not hold but old would (or similar states: others >= 2)
    sel2 = sel & (others >= 2) & (others <= 6)
    idx2 = np.where(sel2)[0]; a2 = (act[idx2] == ph[idx2]).astype(float)
    for K, arr, nm in [(12, Htot, "H12"), (24, Htot, "H24")]:
        e = est(idx2, a2, K, arr)
        print(f"ctl{c} others 2..6: n={len(idx2)} hold share {a2.mean():.3f}  {nm} effect of hold {e[0]:7.1f} ± {e[1]:5.1f}")
m = ok & (ctl == 0) & (g == ph) & (g1 != ph)
print("removed holds in ctl0:", m.sum(), " others H quantiles:", np.quantile(others[m], [0, .25, .5, .75, 1]))
