import sys; sys.path.insert(0, ".")
import numpy as np, tools
S = tools.states()
X = S["obs"].astype(float); act = S["act"].astype(int); rs = S["rule_state"]
ep = S["episode"]; ts = S["time_s"]
w = tools.load("start_weights.npz")
L = tools.forward(w, X)
ss = rs[:, 0:4]; legal = rs[:, 4:8] > 0.5; overdue = rs[:, 8:12] > 0.5; ph = rs[:, 12:16]; tg = rs[:, 16]; g = rs[:, 17].astype(int)
N = len(X)
ill = ~legal[np.arange(N), act]
print("N", N, "illegal requests", ill.sum(), ill.mean())
ok = X[:, 44] > 0.5
print("ok share", ok.mean())
# categories
sw = act != g
print("switch requests", sw.sum(), "with ok", (sw & ok).sum())
# illegal: classify
cat_min = ill & sw & ok & (tg < 15) & np.isin(g, [0, 2])
cat_min2 = ill & sw & (tg < 10)
cat_stay = ill & ~sw
anyover = overdue.any(1)
cat_over = ill & sw & anyover & ~cat_min
print("illegal: switch-before-min-through(ok)", cat_min.sum(), " switch tg<10", cat_min2.sum(),
      " stay (maxgreen)", cat_stay.sum(), " switch with overdue", cat_over.sum())
print("other illegal", (ill & ~cat_min & ~cat_min2 & ~cat_stay & ~cat_over).sum())
# t_green distribution
print("tg values", np.unique(tg, return_counts=True))
# illegal stays: tg distribution
print("illegal stays tg", np.unique(tg[cat_stay], return_counts=True))
print("overdue count states", anyover.sum(), "overdue per phase", overdue.sum(0))
# legal pattern
print("legal patterns", {tuple(r): c for r, c in zip(*np.unique(legal.astype(int), axis=0, return_counts=True))})
