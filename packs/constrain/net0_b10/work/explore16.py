import sys; sys.path.insert(0, ".")
import numpy as np, tools
S = tools.states()
X = S["obs"].astype(float); act = S["act"].astype(int); rs = S["rule_state"]; ep = S["episode"]
w = tools.load("start_weights.npz"); L = tools.forward(w, X)
legal = rs[:, 4:8] > 0.5; od = rs[:, 8:12] > .5; ph = rs[:, 12:16]; tg = rs[:, 16]; g = rs[:, 17].astype(int)
N = len(X); real = legal.sum(1) > 1; tot = ph.sum(1)
bins = [(0, 8), (8, 14), (14, 40)]
print("real decisions per bin", [int((real & (tot >= lo) & (tot < hi)).sum()) for lo, hi in bins])
for D, K in [(2, 0), (4, 0), (3, 3), (4, 3), (4, 4)]:
    ch = act.copy()
    for i in np.where(real)[0]:
        adj = L[i].copy()
        for p in (0, 2):
            if p != g[i] and legal[i, p] and not od[i, p]:
                adj[p] -= D
        # left alternatives must have >= K halting to benefit from the penalty
        cand = [p for p in range(4) if legal[i, p]]
        best = max(cand, key=lambda p: adj[p])
        if best in (1, 3) and best != g[i] and act[i] in (0, 2) and ph[i, best] < K:
            best = act[i]
        ch[i] = best
    f = real & (ch != act)
    print(f"D={D} K={K}: swaps {f.sum()} by bin", [int((f & (tot >= lo) & (tot < hi)).sum()) for lo, hi in bins],
          " left-queue at swaps median", np.median(ph[f, ch[f]]) if f.sum() else None, " through-queue", np.median(ph[f, act[f]]) if f.sum() else None)
