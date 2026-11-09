"""Surrogate check: hide a pocket detector whose truth we know (S2, N2 or E2), impute it a few
principled ways, and measure how often the network's decision matches its full-information decision.
Only forward passes; no fitting."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools

w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float)
ctrl = S["controller"]
ok = X[:, 44] > 0.5

def agree(Xa, Xb, m):
    a = tools.greedy(w, Xa); b = tools.greedy(w, Xb)
    return (a[m] == b[m]).mean()

# pairs: (hidden q, hidden d, mirror q, mirror d, own upstream q, own upstream d)
cases = {
    "hide S2, mirror N2": (8, 20, 2, 14, 29, 37),
    "hide N2, mirror S2": (2, 14, 8, 20, 25, 33),
}
for name, (hq, hd, mq, md, uq, ud) in cases.items():
    print(name)
    for ci in range(3):
        m = (ctrl == ci) & ok
        base = X.copy()
        res = {}
        Z = X.copy(); Z[:, hq] = 0; Z[:, hd] = 0; res["zero"] = agree(Z, base, m)
        mu_q, mu_d = X[ctrl == ci, mq].mean(), X[ctrl == ci, md].mean()
        Z = X.copy(); Z[:, hq] = mu_q; Z[:, hd] = mu_d; res["mean(mirror)"] = agree(Z, base, m)
        Z = X.copy(); Z[:, hq] = X[:, mq]; Z[:, hd] = X[:, md]; res["mirror"] = agree(Z, base, m)
        Z = X.copy(); Z[:, hq] = 0.5 * X[:, mq] + 0.5 * mu_q; Z[:, hd] = 0.5 * X[:, md] + 0.5 * mu_d; res["half mirror+mean"] = agree(Z, base, m)
        print("  ", S["controller_names"][ci], {k: round(v, 3) for k, v in res.items()}, "n", m.sum())
# E2: no mirror available (W2 dead), only own-approach info
print("hide E2 (no mirror)")
for ci in range(3):
    m = (ctrl == ci) & ok
    res = {}
    Z = X.copy(); Z[:, 5] = 0; Z[:, 17] = 0; res["zero"] = agree(Z, X, m)
    Z = X.copy(); Z[:, 5] = X[ctrl == ci, 5].mean(); Z[:, 17] = X[ctrl == ci, 17].mean(); res["mean"] = agree(Z, X, m)
    print("  ", S["controller_names"][ci], {k: round(v, 3) for k, v in res.items()})
