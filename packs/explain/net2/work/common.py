import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states(); X = S['obs'].astype(float)
LG = tools.forward(w, X); A = LG.argmax(1)
PH = X[:, 40:44].argmax(1); OK = X[:, 44] > 0.5; CTRL = S['controller']
LANES = ['N0','N1','N2','E0','E1','E2','S0','S1','S2','W0','W1','W2']
LI = {n: i for i, n in enumerate(LANES)}
def v(X, n):  # lane load in vehicles: halting vehicles count twice (queue + density), moving once
    X = np.atleast_2d(X); return 3*(X[:, LI[n]] + X[:, 12 + LI[n]])
def report(r, mask=None, label=""):
    m = OK if mask is None else mask
    out = {"all_ok": (r[m] == A[m]).mean()}
    for c in range(3): out[f"c{c}"] = (r[m & (CTRL == c)] == A[m & (CTRL == c)]).mean()
    out["per_phase"] = [round((r[m & (PH == p)] == A[m & (PH == p)]).mean(), 3) for p in range(4)]
    print(label, {k: (round(x, 4) if not isinstance(x, list) else x) for k, x in out.items()})
def confusion(r, m):
    M = np.zeros((4, 4), int)
    for i, j in zip(A[m], r[m]): M[i, j] += 1
    return M
