"""Where does the chosen rule differ from the halting-only rule (~ tuned actuated)? Descriptive only."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools
from build import P, score_matrix
S = tools.states(); X = S["obs"].astype(np.float64); C = S["controller"]; A = S["act"]
names = tools.names(); ix = {n: i for i, n in enumerate(names)}
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
np.set_printoptions(precision=2, suppress=True, linewidth=160)
CAP_S = 22.4 / 7.5
Ho = dict(P, WM=0.0, WUM=0.0, K=0.5)
def dec(Pv, Xs):
    Wsc, b = score_matrix(Pv); return (Xs @ Wsc.T + b).argmax(1), Xs @ Wsc.T + b
for name, Pv in [("V1", dict(P)), ("V1+leftprem", dict(P, LEFT_PREM=1.2))]:
    m = ok
    if "LEFT_PREM" in Pv:
        Wsc, b = score_matrix(Pv)
        for p, lanes in ((1, ["N2", "S2"]), (3, ["E2", "W2"])):
            for l in lanes:
                Wsc[p, ix["q_" + l]] += CAP_S * (Pv["LEFT_PREM"] - 1.0)
        d = (X[m] @ Wsc.T + b).argmax(1)
    else:
        d, _ = dec(Pv, X[m])
    h, hs = dec(Ho, X[m])
    M = np.zeros((4, 4), int)
    for i, j in zip(h, d): M[i, j] += 1
    print(name, "rows: halting-only choice, cols: rule choice (ok states, both controllers)\n", M)
    print("  agreement", np.mean(d == h), " choice dist", np.bincount(d, minlength=4) / len(d))
# pocket fill when actuated serves a left phase
m = ok & (C == 1) & (A != ph)
for p, lanes in ((1, ["N2", "S2"]), (3, ["E2", "W2"])):
    mm = m & (A == p)
    fill = sum(CAP_S * X[mm, ix["q_" + l]] for l in lanes)
    print(f"actuated switches to left phase {p}: n {mm.sum()} pocket halting veh mean {fill.mean():.2f} quantiles {np.quantile(fill, [.1, .5, .9])}")
# examples where V1 keeps but halting-only switches
d, sc = dec(P, X[ok]); h, _ = dec(Ho, X[ok])
idx = np.flatnonzero((d == ph[ok]) & (h != ph[ok]))
print("\nV1 keeps, halting-only switches:", len(idx))
for k in idx[:8]:
    x = X[ok][k]; p = ph[ok][k]
    print(f"  cur {p} tg {x[45]*60:.0f}s scores {sc[k]}  cur stop dens {[round(float(x[ix['d_'+l]]*CAP_S),1) for l in {0:['N0','N1','S0','S1'],1:['N2','S2'],2:['E0','E1','W0','W1'],3:['E2','W2']}[p]]}")
