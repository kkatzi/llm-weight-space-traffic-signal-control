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
N = len(X)
free = legal.all(1) & ~overdue.any(1)
print("free opportunities", free.sum())
THRU = (0, 2); LEFT = (1, 3)
for d in (0.25, 0.5, 0.75, 1.0, 1.5):
    n = 0
    for i in np.where(free)[0]:
        a = act[i]
        if a in THRU:
            l = max(LEFT, key=lambda p: lg[i, p])
            if l != green[i] and lg[i, a] - lg[i, l] < d:
                n += 1
    print("through top, best left within", d, ":", n, f"({n/free.sum():.3f})")
# distribution of top-2 gap at free opp
srt = np.sort(lg, 1); gap = srt[:, -1] - srt[:, -2]
print("free gap quantiles", np.quantile(gap[free], [.05, .1, .2, .3, .5]))
# logit sanity: correlation of logits with halting per phase at free opp
for p in range(4):
    print(p, "corr(logit_p - mean, halt_p)", np.corrcoef(lg[free, p] - lg[free].mean(1), halt[free, p])[0, 1].round(3))
