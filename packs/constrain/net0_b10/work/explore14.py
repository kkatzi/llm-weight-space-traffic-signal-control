import sys; sys.path.insert(0, ".")
import numpy as np, tools
S = tools.states()
X = S["obs"].astype(float); act = S["act"].astype(int); rs = S["rule_state"]; ep = S["episode"]
w = tools.load("start_weights.npz")
L = tools.forward(w, X)
legal = rs[:, 4:8] > 0.5; ph = rs[:, 12:16]; tg = rs[:, 16]; g = rs[:, 17].astype(int)
N = len(X); real = legal.sum(1) > 1
I = {n: i for i, n in enumerate(tools.names())}
pairs = {2: (3, ["E", "W"]), 0: (1, ["N", "S"])}
for p, (a, apps) in pairs.items():
    m = real & (act == p) & (g != p)
    spill = np.zeros(N, bool)
    for ap in apps:
        spill |= (X[:, I[f"q_{ap}2"]] > 0.99) & (X[:, I[f"q_{ap}1u"]] > 0.02)
    mm = m & spill
    gap = L[np.arange(N), p] - L[np.arange(N), a]
    print(f"through {p} chosen at real decisions: {m.sum()}, with its left pocket spilling: {mm.sum()}, gap to left {a}: median {np.median(gap[mm]) if mm.sum() else 0:.2f}, share<2 {(gap[mm]<2).mean() if mm.sum() else 0:.2f}")
