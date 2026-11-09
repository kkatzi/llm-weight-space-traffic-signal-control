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
LANES = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
UP = {0: [24, 25, 28, 29], 1: [25, 29], 2: [26, 27, 30, 31], 3: [27, 31]}
ok = X[:, 44] > 0.5
alllegal = legal.all(1)
# decision points that matter: ok & more than one legal
dec = ok & (legal.sum(1) > 1)
print("real decision points", dec.sum())
sw = act != g
print("of which network switches", (dec & sw).sum(), " holds", (dec & ~sw).sum())
# through phases at tg=12, network wants switch: density of current lanes
m = ok & (tg == 12) & np.isin(g, [0, 2]) & sw
dq = np.array([X[i, [12 + l for l in LANES[g[i]]]].sum() for i in np.where(m)[0]])
qq = np.array([X[i, LANES[g[i]]].sum() for i in np.where(m)[0]])
print("forced through hold at 12: n", m.sum(), "sum density cur lanes mean", dq.mean(), "quantiles", np.quantile(dq, [.1, .25, .5, .75, .9]))
print("  halting cur lanes (obs sum)", np.quantile(qq, [.1, .25, .5, .75, .9]))
print("  phase_halting cur", np.quantile(ph[m, :][np.arange(m.sum()), g[m]], [.1, .5, .9]))
# at tg=17 for those: what does the network choose
m17 = ok & (tg == 17) & np.isin(g, [0, 2])
print("through tg=17: n", m17.sum(), "hold share", (act[m17] == g[m17]).mean())
m12l = ok & (tg == 12) & np.isin(g, [1, 3])
print("left tg=12: n", m12l.sum(), "hold share", (act[m12l] == g[m12l]).mean())
# chosen phase distribution at real switch decisions
d = dec & sw
print("next phase chosen from each g:")
for gg in range(4):
    mm = d & (g == gg)
    print(gg, np.bincount(act[mm], minlength=4))
# phase transitions actual sequence & green durations
# derive green durations per episode
durs = {p: [] for p in range(4)}
for e in np.unique(ep):
    idx = np.where(ep == e)[0]
    for k in range(len(idx) - 1):
        i, j = idx[k], idx[k + 1]
        if g[j] != g[i] or (tg[j] < tg[i]):
            durs[g[i]].append(tg[i] + 5 - 3)  # green time approx
for p in range(4):
    print("phase", p, "n greens", len(durs[p]), "dur dist", np.unique(durs[p], return_counts=True))
