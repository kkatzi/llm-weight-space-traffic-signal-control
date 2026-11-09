import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load("start_weights.npz")
S = tools.states(); X = S['obs'].astype(float); a = tools.greedy(w, X)
names = tools.names()
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
def grads(X):
    a1 = X @ w["W1"].T + w["b1"]; h1 = np.tanh(a1)
    a2 = h1 @ w["W2"].T + w["b2"]; h2 = np.tanh(a2)
    return np.einsum('ok,nk,kj,nj,ji->noi', w["W3"], 1-h2**2, w["W2"], 1-h1**2, w["W1"])
lanes = ['N0','N1','N2','E0','E1','E2','S0','S1','S2','W0','W1','W2']
for p in range(4):
    m = ok & (ph == p)
    G = grads(X[m]).mean(0)
    cands = [j for j in range(4) if j != p]
    print(f"=== current phase {p}: per halting vehicle (q+d)/3, and per moving vehicle d/3")
    print("            " + " ".join(f"{l:>5s}" for l in lanes) + " | moving: " + " ".join(f"{l:>5s}" for l in lanes))
    for i in cands:
        for j in cands:
            if j <= i: continue
            g = G[i] - G[j]
            hv = (g[0:12] + g[12:24])/3; mv = g[12:24]/3
            print(f"L{i}-L{j}      " + " ".join(f"{v:5.2f}" for v in hv) + " |         " + " ".join(f"{v:5.2f}" for v in mv))
