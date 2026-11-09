import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load("start_weights.npz")
S = tools.states(); X = S['obs'].astype(float); a = tools.greedy(w, X)
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
def groups(X):
    # vehicles (density*3) per group, halting (queue*3)
    d = X[:, 12:24]*3; q = X[:, 0:12]*3
    G = {}
    for nm, v in [('d', d), ('q', q)]:
        G[nm+'NS_T'] = v[:, [0,1,6,7]].sum(1); G[nm+'NS_L'] = v[:, [2,8]].sum(1)
        G[nm+'EW_T'] = v[:, [3,4,9,10]].sum(1); G[nm+'EW_L'] = v[:, [5,11]].sum(1)
    G['upNS'] = X[:, [32,33,36,37]].sum(1)*34; G['upEW'] = X[:, [34,35,38,39]].sum(1)*34
    return G
G = groups(X)
keys = list(G.keys())
print("phase act n | " + " ".join(f"{k:>6s}" for k in keys))
for p in range(4):
    for j in range(4):
        m = ok & (ph == p) & (a == j)
        if m.sum() < 5: continue
        print(f"{p} {j} {m.sum():4d} | " + " ".join(f"{G[k][m].mean():6.2f}" for k in keys))
