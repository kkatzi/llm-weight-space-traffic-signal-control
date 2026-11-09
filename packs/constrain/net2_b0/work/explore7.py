import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
from collections import Counter
np.set_printoptions(precision=2, suppress=True, linewidth=220)
S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
lg = tools.forward(w, X)
act = S["act"].astype(int)
rs = S["rule_state"]
since = rs[:, 0:4]; legal = rs[:, 4:8] > 0.5; overdue = rs[:, 8:12] > 0.5; halt = rs[:, 12:16]
tg = rs[:, 16]; green = rs[:, 17].astype(int)
ep = S["episode"]; t = S["time_s"]
N = len(X)
opp = (tg >= np.array([15, 10, 15, 10])[green]) & ~overdue.any(1)
srt = np.sort(lg, 1); gap = srt[:, -1] - srt[:, -2]
for p in range(4):
    m = opp & (act == p)
    print("phase", p, "chosen", m.sum(), "halting dist", sorted(Counter(halt[m, p].astype(int)).items()))
    print("    gap when chosen quantiles", np.quantile(gap[m], [.1, .25, .5]))
# upstream queue capacity: 256/7.5 = 34.1 per lane
cap_up = 256 / 7.5
print("upstream q stats (vehicles):")
for j, n in enumerate(tools.names()[24:32]):
    v = X[:, 24 + j] * cap_up
    print(n, "mean", v.mean().round(2), "p90", np.quantile(v, .9).round(2), "max", v.max().round(2))
cap = 22.4 / 7.5
print("stop-line q (vehicles) share at full:")
for j, n in enumerate(tools.names()[0:12]):
    print(n, "mean", (X[:, j] * cap).mean().round(2), "full share", (X[:, j] >= 0.99).mean().round(3))
# by episode time bucket: mean total halting
tb = (t // 900).clip(0, 4)
for b in range(5):
    m = tb == b
    print("bucket", b, "n", m.sum(), "mean halt per phase", halt[m].mean(0).round(2), "mean up q total", (X[m, 24:32].sum(1) * cap_up).mean().round(2))
