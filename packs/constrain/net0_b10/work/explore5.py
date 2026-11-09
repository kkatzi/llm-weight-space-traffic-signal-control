import sys; sys.path.insert(0, ".")
import numpy as np, tools
S = tools.states()
X = S["obs"].astype(float); act = S["act"].astype(int); rs = S["rule_state"]
ep = S["episode"]; ts = S["time_s"]
w = tools.load("start_weights.npz")
L = tools.forward(w, X)
ss = rs[:, 0:4]; legal = rs[:, 4:8] > 0.5; overdue = rs[:, 8:12] > 0.5; ph = rs[:, 12:16]; tg = rs[:, 16]; g = rs[:, 17].astype(int)
N = len(X)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
ok = X[:, 44] > 0.5
dec = ok & legal.all(1)
sw = act != g
d = np.where(dec & sw)[0]
# for each switch decision, find the next state (5 s later) index and the green outcome
nxt = {}
for e in np.unique(ep):
    idx = np.where(ep == e)[0]
    for k in range(len(idx) - 1):
        nxt[idx[k]] = idx[k + 1]
thr = [i for i in d if act[i] in (0, 2)]
lef = [i for i in d if act[i] in (1, 3)]
print("switch to through:", len(thr), " to left:", len(lef))
hp = np.array([ph[i, act[i]] for i in thr])
print("halting on chosen through phase at switch: quantiles", np.quantile(hp, [.1, .25, .5, .75, .9]))
hl = np.array([ph[i, act[i]] for i in lef])
print("halting on chosen left phase at switch: quantiles", np.quantile(hl, [.1, .25, .5, .75, .9]))
# logit gap top vs 2nd for through choices, and 2nd phase identity
srt = np.argsort(-L, axis=1)
gap = L[np.arange(N), srt[:, 0]] - L[np.arange(N), srt[:, 1]]
print("through choice gap quantiles", np.quantile(gap[thr], [.1, .25, .5, .75, .9]))
print("2nd choice when through chosen", np.bincount(srt[thr, 1], minlength=4))
# total halting elsewhere at switch to through
oth = np.array([ph[i].sum() - ph[i, act[i]] for i in thr])
print("halting on other phases at through switch", np.quantile(oth, [.1, .5, .9]))
# relation: halting on chosen through vs gap
for lo, hi in [(0, 1), (1, 3), (3, 5), (5, 8), (8, 100)]:
    m = (hp >= lo) & (hp < hi)
    print(f"hp in [{lo},{hi}): n={m.sum()}, median gap {np.median(gap[np.array(thr)[m]]) if m.sum() else 0:.2f}, share gap<1 {(gap[np.array(thr)[m]]<1).mean() if m.sum() else 0:.2f}")
# Also total halting at decision (all phases), as demand indicator
print("phase_halting sums at decisions quantiles", np.quantile(ph[dec].sum(1), [.1, .5, .9]))
