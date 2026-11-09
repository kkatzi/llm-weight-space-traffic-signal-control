import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
lg = tools.forward(w, X); g = lg.argmax(1)
r = np.arange(len(X))
s = np.sort(lg, 1); gap = s[:, -1] - s[:, -2]
m = ok & (ctrl == 0)
print("honoured net decisions: gap quantiles", np.quantile(gap[m], [0.05, 0.1, 0.25, 0.5]).round(2))
print("frac gap<0.5:", (gap[m] < 0.5).mean(), " <1:", (gap[m] < 1).mean(), " <2:", (gap[m] < 2).mean())
# stays: margin of current phase over best other
lgo = lg.copy(); lgo[r, ph] = -np.inf
stay_margin = lg[r, ph] - lgo.max(1)
mm = m & (g == ph)
print("stay margins:", np.sort(stay_margin[mm]).round(2))
# what does the stay margin look like per phase at honoured decisions overall
for p in range(4):
    mm = ok & (ph == p)
    print(f"phase {p}: honoured n={mm.sum()} stay-margin quantiles", np.quantile(stay_margin[mm], [0.5, 0.9, 0.95, 0.99, 1]).round(2), "stays", (stay_margin[mm] > 0).sum())
# effect of b3 shifts: number of honoured decisions changed for shift of +d on each logit
for p in range(4):
    for dlt in [-1, -0.5, 0.5, 1]:
        w2 = {k: v.copy() for k, v in w.items()}; w2["b3"][p] += dlt
        g2 = tools.greedy(w2, X)
        print(f" b3[{p}] {dlt:+}: honoured changed {(g2[ok] != g[ok]).mean():.3f} (net eps {(g2[m] != g[m]).mean():.3f}); share picks p in net eps {(g2[m]==p).mean():.3f} vs {(g[m]==p).mean():.3f}")
