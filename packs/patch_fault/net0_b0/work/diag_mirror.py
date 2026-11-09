"""Diagnostic only (no weights are fitted): on the N/S and E sides, where the left pockets ARE observed,
hide one pocket and fill it with a hand-chosen imputation; measure how often the start network's decision
is the same as with the true value. Used to choose between a few candidate imputation rules."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools

w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float)
ok = X[:, 44] > 0.5
n = tools.names(); I = n.index
true = tools.greedy(w, X)


def agree(Y):
    a = tools.greedy(w, Y)
    return (a[ok] == true[ok]).mean(), (a == true).mean()


def run(target, proxy, upq, label):
    tq, td = I("q_" + target), I("d_" + target)
    print(label)
    cands = {}
    Y = X.copy(); Y[:, tq] = 0; Y[:, td] = 0; cands["zero"] = Y
    m = X[:, tq].mean(); md = X[:, td].mean()
    Y = X.copy(); Y[:, tq] = m; Y[:, td] = md; cands["mean"] = Y
    if proxy:
        pq, pd = I("q_" + proxy), I("d_" + proxy)
        Y = X.copy(); Y[:, tq] = X[:, pq]; Y[:, td] = X[:, pd]; cands["proxy"] = Y
        for a in [0.5, 0.75]:
            Y = X.copy(); Y[:, tq] = a * X[:, pq] + (1 - a) * m; Y[:, td] = a * X[:, pd] + (1 - a) * md
            cands[f"{a}*proxy+{1-a}*mean"] = Y
        for k in [2, 3, 5]:
            Y = X.copy(); Y[:, tq] = X[:, pq] + k * X[:, I(upq)]; Y[:, td] = X[:, pd] + k * X[:, I(upq)]
            cands[f"proxy+{k}*{upq}"] = Y
    for k in [2, 3, 5]:
        Y = X.copy(); Y[:, tq] = k * X[:, I(upq)]; Y[:, td] = k * X[:, I(upq)]
        cands[f"{k}*{upq} only"] = Y
    for name, Y in cands.items():
        h, a = agree(Y)
        print(f"   {name:28s} agreement honoured={h:.4f} all={a:.4f}")


run("S2", "N2", "q_S1u", "hide S2, proxy N2")
run("N2", "S2", "q_N1u", "hide N2, proxy S2")
run("E2", None, "q_E1u", "hide E2 (no proxy available, W2 dead)")
