import sys; sys.path.insert(0, ".")
import numpy as np, tools
S = tools.states()
X = S["obs"].astype(float); act = S["act"].astype(int); rs = S["rule_state"]; ep = S["episode"]; ts = S["time_s"]
w = tools.load("start_weights.npz")
L = tools.forward(w, X)
legal = rs[:, 4:8] > 0.5; od = rs[:, 8:12] > .5; ph = rs[:, 12:16]; tg = rs[:, 16]; g = rs[:, 17].astype(int); ss = rs[:, 0:4]
N = len(X); ok = X[:, 44] > 0.5
real = legal.sum(1) > 1
rows = []
for i in np.where(real)[0]:
    top = act[i]
    if top == g[i]:
        T = 5
    else:
        T = 20 if top in (0, 2) else 15
    for p in range(4):
        if p == top or p == g[i] or not legal[i, p]:
            continue
        if ph[i, p] >= 1 and ss[i, p] + T >= 90:
            rows.append((i, p, L[i, top] - L[i, p], T))
rows = np.array(rows)
print("imminent-overdue real decisions:", len(rows), "(unique states", len(np.unique(rows[:, 0])), ")")
print("gap quantiles", np.quantile(rows[:, 2], [.1, .25, .5, .75, .9]).round(2))
for M in [0.5, 1, 2, 3]:
    print(f"  gap<{M}: {(rows[:, 2] < M).sum()}")
# check: does the forced service actually follow in those cases? next real decision legal pattern
nf = 0
for i, p, gap, T in rows:
    i = int(i); p = int(p)
    j = i + int(T // 5)
    if j < N and ep[j] == ep[i]:
        nf += legal[j].sum() == 1 and legal[j, p]
print("followed by forced service of p (where top was taken):", nf)
