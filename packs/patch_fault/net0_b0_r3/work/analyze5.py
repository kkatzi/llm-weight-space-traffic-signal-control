"""(1) E-mirror: pocket level given the number of halted cars on the upstream lane 1, per phase.
(2) Layer-1 unit importance: decisions changed when one unit is frozen at its mean (to find a unit to repurpose).
(3) Network response to W2 values above 1 (extrapolation check)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools

np.set_printoptions(precision=3, suppress=True, linewidth=160)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64)
ctrl = S["controller"]
nm = tools.names()
ix = {n: i for i, n in enumerate(nm)}
ph = X[:, 40:44].argmax(1)
ok = X[:, 44] > 0.5
cap_u = 256 / 7.5

print("(1) E[q_P2], E[d_P2] given halted cars n on P1u (n = round(q_P1u*34.13)), phases != served-left phase")
for P, lp in [("E", 3), ("N", 1), ("S", 1)]:
    n = np.round(X[:, ix[f"q_{P}1u"]] * cap_u).astype(int)
    for p_set, lab in [([p for p in range(4) if p != lp], "not-served"), ([lp], "served")]:
        m = np.isin(ph, p_set)
        row = []
        for k in [0, 1, 2, 3, 4]:
            mm = m & ((n == k) if k < 4 else (n >= k))
            if mm.sum():
                row.append(f"n{'>=' if k == 4 else '='}{k}: q={X[mm, ix[f'q_{P}2']].mean():.2f} "
                           f"d={X[mm, ix[f'd_{P}2']].mean():.2f} ({mm.sum()})")
        print(f"  {P} {lab:10s} " + " | ".join(row))

print("\n(2) layer-1 units: share of honoured decisions changed when unit frozen at its mean")
h1, h2 = tools.hidden(w, X)
a0 = tools.greedy(w, X)
imp = []
for u in range(64):
    hh = h1.copy()
    hh[:, u] = h1[:, u].mean()
    lg = np.tanh(hh @ w["W2"].T + w["b2"]) @ w["W3"].T + w["b3"]
    imp.append((lg.argmax(1)[ok] != a0[ok]).mean())
imp = np.array(imp)
order = np.argsort(imp)
for u in order[:12]:
    print(f"  unit {u}: changed {imp[u]:.4f}  std {h1[:, u].std():.3f} mean {h1[:, u].mean():+.3f} "
          f"|W1[u,11]|={abs(w['W1'][u, 11]):.2f} |W1[u,23]|={abs(w['W1'][u, 23]):.2f}")
print("  trainability", tools.trainability(w))

print("\n(3) decision shares (honoured states) when q_W2 = d_W2 = v")
for v in [0, 0.5, 1, 1.5, 2, 3, 5]:
    Z = X.copy(); Z[:, 11] = v; Z[:, 23] = v
    a = tools.greedy(w, Z)
    print(f"  v={v}: {np.bincount(a[ok], minlength=4) / ok.sum()}")
