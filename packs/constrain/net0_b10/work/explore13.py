import sys; sys.path.insert(0, ".")
import numpy as np, tools
S = tools.states()
X = S["obs"].astype(float); act = S["act"].astype(int); rs = S["rule_state"]; ep = S["episode"]
w = tools.load("start_weights.npz")
L = tools.forward(w, X)
legal = rs[:, 4:8] > 0.5; od = rs[:, 8:12] > .5; ph = rs[:, 12:16]; tg = rs[:, 16]; g = rs[:, 17].astype(int); ss = rs[:, 0:4]
N = len(X); real = legal.sum(1) > 1
tot = ph.sum(1)
# logit gap between through top and best left, by total halting level
srt = np.argsort(-L, 1)
thrtop = real & np.isin(act, [0, 2]) & (act != g)
bl = np.array([max((1, 3), key=lambda a: L[i, a]) for i in range(N)])
gapL = L[np.arange(N), act] - L[np.arange(N), bl]
for lo, hi in [(0, 6), (6, 10), (10, 14), (14, 40)]:
    m = thrtop & (tot >= lo) & (tot < hi)
    print(f"total halting [{lo},{hi}): through-top n={m.sum()}, gap to best left median {np.median(gapL[m]):.2f}, share<2 {(gapL[m]<2).mean():.2f}, share<4 {(gapL[m]<4).mean():.2f}")
    m2 = m & (gapL < 4)
    if m2.sum():
        print(f"     among gap<4: halting thr {np.median(ph[m2, act[m2]]):.1f}, best-left {np.median(ph[m2, bl[m2]]):.1f}")
# cmu tie-break: swap to best left a if gap < M and ph[a]*20 > ph[p]*15
for M in [2, 3, 4, 100]:
    sw = thrtop & (gapL < M) & (ph[np.arange(N), bl] * 20 > ph[np.arange(N), act] * 15) & legal[np.arange(N), bl]
    print(f"M={M}: swaps {sw.sum()}  by total-halting bins", [int((sw & (tot >= lo) & (tot < hi)).sum()) for lo, hi in [(0, 6), (6, 10), (10, 14), (14, 40)]])
print("real through-top decisions per bin", [int((thrtop & (tot >= lo) & (tot < hi)).sum()) for lo, hi in [(0, 6), (6, 10), (10, 14), (14, 40)]])
