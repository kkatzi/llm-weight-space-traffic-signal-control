import sys; sys.path.insert(0, ".")
import numpy as np, tools
S = tools.states()
X = S["obs"].astype(float); act = S["act"].astype(int); rs = S["rule_state"]; ep = S["episode"]
w = tools.load("start_weights.npz"); L = tools.forward(w, X)
legal = rs[:, 4:8] > 0.5; ph = rs[:, 12:16]; tg = rs[:, 16]; g = rs[:, 17].astype(int)
N = len(X); ok = X[:, 44] > .5
m = np.where(ok & (tg == 12) & np.isin(g, [0, 2]))[0]
gap = L[m].max(1) - L[m, g[m]]
print("through tg=12 (forced hold): gap top - hold quantiles", np.quantile(gap, [.1, .25, .5, .75, .9]).round(2))
m2 = np.where(ok & (tg == 12) & np.isin(g, [1, 3]))[0]
gap2 = L[m2].max(1) - L[m2, g[m2]]
print("left tg=12 (free): gap top - hold quantiles", np.quantile(gap2, [.1, .25, .5, .75, .9]).round(2))
m3 = np.where(ok & (tg == 17) & np.isin(g, [0, 2]))[0]
gap3 = L[m3].max(1) - L[m3, g[m3]]
print("through tg=17 (free): gap top - hold quantiles", np.quantile(gap3, [.1, .25, .5, .75, .9]).round(2))
