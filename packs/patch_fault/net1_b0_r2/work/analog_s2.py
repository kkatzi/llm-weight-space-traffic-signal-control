"""Analogue experiment: the S2 detector works in the data. Pretend it failed and see which substitute for
(q_S2, d_S2) keeps the network's decisions closest to its decisions with the true S2 values. Used only to choose
the substitution rule for the real failure (W2) by analogy. No fitting: a handful of hand-picked candidates."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools

S = tools.states(); X0 = S["obs"].astype(float); c = S["controller"]
w = tools.load("start_weights.npz")
# base states: W2 := E2 (our best guess for the actually-missing lane), so the inputs look in-distribution
X = X0.copy(); X[:, 11] = X0[:, 5]; X[:, 23] = X0[:, 17]
ok = X[:, 44] > 0.5
ref = tools.greedy(w, X)

def cand(name, q, d):
    Z = X.copy(); Z[:, 8] = q; Z[:, 20] = d
    g = tools.greedy(w, Z)
    row = [name]
    for ci in range(3):
        m = (c == ci) & ok
        row.append((g[m] == ref[m]).mean())
    m = ok
    row.append((g[m] == ref[m]).mean())
    print("%-28s start %.3f fixed %.3f act %.3f | all %.3f" % tuple(row))

mq, md = X[:, 8].mean(), X[:, 20].mean()
cand("zero (fault)", 0, 0)
cand("mean", mq, md)
cand("N2 proxy", X[:, 2], X[:, 14])
cand("0.5 N2 + 0.5 mean", 0.5 * X[:, 2] + 0.5 * mq, 0.5 * X[:, 14] + 0.5 * md)
cand("0.7 N2 + 0.3 mean", 0.7 * X[:, 2] + 0.3 * mq, 0.7 * X[:, 14] + 0.3 * md)
cand("N2 + S1u", np.clip(X[:, 2] + X[:, 29], 0, 1), np.clip(X[:, 14] + X[:, 37], 0, 1))
cand("0.5N2+0.5mean+S1u", 0.5 * X[:, 2] + 0.5 * mq + X[:, 29], 0.5 * X[:, 14] + 0.5 * md + X[:, 37])
cand("S1 proxy", X[:, 7], X[:, 19])
cand("0.5(N2+S1)", 0.5 * (X[:, 2] + X[:, 7]), 0.5 * (X[:, 14] + X[:, 19]))
cand("mean of N2,S0,S1", (X[:, 2] + X[:, 6] + X[:, 7]) / 3, (X[:, 14] + X[:, 18] + X[:, 19]) / 3)
cand("full (1,1)", 1, 1)
