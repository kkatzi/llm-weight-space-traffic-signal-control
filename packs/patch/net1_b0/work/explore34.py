import sys; sys.path.insert(0, '.')
import numpy as np, tools
src = open('work/explore33.py').read().split("for c in [0, 2]:")[0]
exec(src)
lgs = np.sort(lg, 1)
# gap between network's top choice and the index's choice logit (how much the network dislikes the index choice)
for G in [10]:
    s = choice(H - G*lam)
    gap_idx = lg[I, g] - lg[I, s]   # >=0, 0 if agree
    for c in [0, 2]:
        sel = ok & (act != ph) & (t < 3300) & (ctl == c)
        for lo, hi in [(0, 1.0), (0, 2.0), (0, 4.0), (0, 1e9)]:
            # restrict to states where either agree, or disagree with gap in [lo,hi)
            m = sel & ((gap_idx == 0) | ((gap_idx > lo) & (gap_idx < hi)))
            idx = np.where(m)[0]
            a = (act[idx] == s[idx]).astype(float)
            e1 = est(idx, a, 24, Ntot); e2 = est(idx, a, 12, Htot); e3 = est(idx, a, 24, Htot)
            print(f"ctl{c} disagreements with netgap<{hi}: n_dis={(a==0).sum():4d}  N K24 {e1[0]:7.1f}±{e1[1]:5.1f}  H K12 {e2[0]:7.1f}±{e2[1]:5.1f}  H K24 {e3[0]:7.1f}±{e3[1]:5.1f}")
