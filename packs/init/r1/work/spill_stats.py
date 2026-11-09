"""When the shared upstream lane 1 has halting vehicles, which stop-line lane is full? (descriptive)"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools
from build import P, rule_scores

S = tools.states(); X = S["obs"].astype(np.float64); C = S["controller"]; A = S["act"]
names = tools.names(); ix = {n: i for i, n in enumerate(names)}
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
np.set_printoptions(precision=3, suppress=True, linewidth=160)
for ap in "NESW":
    qu = X[:, ix[f"q_{ap}1u"]]; q1 = X[:, ix[f"q_{ap}1"]]; q2 = X[:, ix[f"q_{ap}2"]]; q0u = X[:, ix[f"q_{ap}0u"]]
    m = qu > 0
    print(f"{ap}: n(q_{ap}1u>0)={m.sum():4d}  pocket full {np.mean(q2[m] > .99):.2f}  straight full {np.mean(q1[m] > .99):.2f}"
          f"  both {np.mean((q2[m] > .99) & (q1[m] > .99)):.2f}  pocket-full-only {np.mean((q2[m] > .99) & (q1[m] < .99)):.2f}"
          f"  straight-full-only {np.mean((q2[m] < .99) & (q1[m] > .99)):.2f}  q_{ap}0u>0 too {np.mean(q0u[m] > 0):.2f}")
    # conditional on which phase is green
    for p in range(4):
        mm = m & (ph == p)
        if mm.sum() > 10:
            print(f"     green {p}: n {mm.sum():4d} pocket-full-only {np.mean((q2[mm] > .99) & (q1[mm] < .99)):.2f}"
                  f" straight-full-only {np.mean((q2[mm] < .99) & (q1[mm] > .99)):.2f} both {np.mean((q2[mm] > .99) & (q1[mm] > .99)):.2f}"
                  f" mean q1u veh {34.13 * qu[mm].mean():.1f}")

# disagreements with actuated at ok states when both switch
sc = rule_scores(P, X); a = sc.argmax(1)
m = ok & (C == 1) & (A != ph) & (a != ph) & (a != A)
print("\nactuated switch targets vs mine (rows: actuated, cols: mine):")
M = np.zeros((4, 4), int)
for i, j in zip(A[m], a[m]): M[i, j] += 1
print(M)
CAP_S = 22.4 / 7.5
def halt(x, p):
    L = {0: ["N0", "N1", "S0", "S1"], 1: ["N2", "S2"], 2: ["E0", "E1", "W0", "W1"], 3: ["E2", "W2"]}[p]
    return sum(CAP_S * x[ix["q_" + l]] for l in L)
for k in np.flatnonzero(m)[:12]:
    x = X[k]
    print(f"cur {ph[k]} act {A[k]} mine {a[k]} halting {[round(halt(x, p), 1) for p in range(4)]} scores {sc[k]}")
