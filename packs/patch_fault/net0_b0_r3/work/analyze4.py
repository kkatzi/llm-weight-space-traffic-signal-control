"""Conditional means of a left pocket given (a) the opposite pocket and (b) its own upstream lane 1.
Uses the working pockets (N2, S2, E2) as mirrors for the failed W2. Statistics only."""
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
ph = X[:, 40:44].argmax(1)

print("E[q_S2 | q_N2] and E[q_N2 | q_S2] (opposite pockets, same phase), all data")
for v in [0, 0.335, 0.67, 1.0]:
    m1 = np.isclose(X[:, ix["q_N2"]], v, atol=0.01)
    m2 = np.isclose(X[:, ix["q_S2"]], v, atol=0.01)
    print(f"  opp={v:.2f}: E[q_S2|q_N2]={X[m1, ix['q_S2']].mean():.3f} (n={m1.sum()})  "
          f"E[q_N2|q_S2]={X[m2, ix['q_N2']].mean():.3f} (n={m2.sum()})")

print("\nE[q_P2 | d_P1u bin] for P in E,N,S, by controller data")
bins = [0, 0.03, 0.06, 0.09, 0.12, 0.18, 0.3, 0.5, 1.01]
for P in ["E", "N", "S"]:
    for c in range(3):
        m = ctrl == c
        d1u = X[m, ix[f"d_{P}1u"]]
        q2 = X[m, ix[f"q_{P}2"]]
        row = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = (d1u >= lo) & (d1u < hi)
            row.append(f"[{lo:.2f},{hi:.2f}):{q2[mm].mean() if mm.sum() else float('nan'):.2f}({mm.sum()})")
        print(f"  {P} ctrl{c}: " + " ".join(row))

print("\nE[q_P2 | q_P1u>0] vs E[q_P2 | q_P1u==0], by phase (all data)")
for P in ["E", "N", "S"]:
    for p in range(4):
        m = ph == p
        q1u = X[m, ix[f"q_{P}1u"]]
        q2 = X[m, ix[f"q_{P}2"]]
        a, b = q1u > 0, q1u == 0
        print(f"  {P} phase{p}: q1u>0: {q2[a].mean():.2f} (n={a.sum()})  q1u==0: {q2[b].mean():.2f} (n={b.sum()})")

print("\ndistribution of d_W1u vs d_E1u, q_W1u vs q_E1u per controller (quantiles)")
for c in range(3):
    m = ctrl == c
    for f in ["d_W1u", "d_E1u", "q_W1u", "q_E1u", "d_W1", "d_E1", "q_W1", "q_E1"]:
        print(f"  ctrl{c} {f}: q50={np.quantile(X[m, ix[f]], .5):.3f} q90={np.quantile(X[m, ix[f]], .9):.3f} "
              f"q99={np.quantile(X[m, ix[f]], .99):.3f} max={X[m, ix[f]].max():.3f}")
