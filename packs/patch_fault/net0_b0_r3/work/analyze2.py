"""Which observed inputs carry information about the (unobserved) W2 pocket?

W2 itself is never observed (always 0 in the data). E and W have identical demand, and N and S have identical
demand, so the mirror lanes tell us how a left pocket relates to (a) the opposite pocket served in the same phase
and (b) its own approach's upstream lane. Only correlations / means are computed here (no fitting).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools

np.set_printoptions(precision=3, suppress=True, linewidth=160)
S = tools.states()
X = S["obs"].astype(np.float64)
ctrl = S["controller"]
nm = tools.names()
ix = {n: i for i, n in enumerate(nm)}

def corr(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

for c, cname in enumerate(S["controller_names"]):
    m = ctrl == c
    Xm = X[m]
    print(f"\n=== data from controller {cname}  n={m.sum()}")
    print("  opposite pocket, same phase: corr(q_N2,q_S2)=%.3f corr(d_N2,d_S2)=%.3f" %
          (corr(Xm[:, ix['q_N2']], Xm[:, ix['q_S2']]), corr(Xm[:, ix['d_N2']], Xm[:, ix['d_S2']])))
    for P in ["E", "N", "S"]:
        q2, d2 = Xm[:, ix[f"q_{P}2"]], Xm[:, ix[f"d_{P}2"]]
        row = []
        for f in [f"q_{P}1u", f"d_{P}1u", f"q_{P}1", f"d_{P}1", f"q_{P}0", f"d_{P}0", "t_green"]:
            row.append(f"{f}:{corr(q2, Xm[:, ix[f]]):+.2f}/{corr(d2, Xm[:, ix[f]]):+.2f}")
        print(f"  {P}2 (q/d) vs own lanes:", "  ".join(row))
    # pocket stats by phase
    ph = Xm[:, 40:44].argmax(1)
    for P in ["N", "E", "S", "W"]:
        s = []
        for p in range(4):
            mm = ph == p
            s.append(f"ph{p}: q={Xm[mm, ix[f'q_{P}2']].mean():.2f} d={Xm[mm, ix[f'd_{P}2']].mean():.2f} "
                     f"q1u={Xm[mm, ix[f'q_{P}1u']].mean():.3f} d1u={Xm[mm, ix[f'd_{P}1u']].mean():.3f}")
        print(f"  {P}: " + " | ".join(s))
    # distribution of pocket values
    for P in ["E", "W"]:
        print(f"  q_{P}2 values:", np.unique(np.round(Xm[:, ix[f'q_{P}2']], 3), return_counts=True))
        print(f"  d_{P}2 values:", np.unique(np.round(Xm[:, ix[f'd_{P}2']], 3), return_counts=True))
