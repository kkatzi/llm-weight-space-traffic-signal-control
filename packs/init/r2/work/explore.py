import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=200, precision=3, suppress=True)
S = tools.states()
X, A, E, T, C = S["obs"].astype(float), S["act"], S["episode"], S["time_s"], S["controller"]
N = tools.names()
print("controllers", S["controller_names"])
for c in range(2):
    m = C == c
    eps = np.unique(E[m])
    print("ctrl", c, "episodes", eps, "lengths", [int((m & (E == e)).sum()) for e in eps], "max t", [int(T[m & (E == e)].max()) for e in eps])
print("\nfeature stats (mean, q50, q90, q99, max)")
for i, n in enumerate(N):
    v = X[:, i]
    print(f"{i:2d} {n:8s} {v.mean():.3f} {np.quantile(v,.5):.3f} {np.quantile(v,.9):.3f} {np.quantile(v,.99):.3f} {v.max():.3f}")
# current phase
ph = X[:, 40:44].argmax(1)
ok = X[:, 44] > 0.5
for c in range(2):
    m = C == c
    print("ctrl", c, "phase dist", np.bincount(ph[m], minlength=4) / m.sum(), "act dist", np.bincount(A[m], minlength=4) / m.sum())
    print("  ok share", ok[m].mean(), " switch share among ok", (A[m & ok] != ph[m & ok]).mean())
    # green durations
    tg = X[m, 45] * 60
    print("  t_green at switch (ok):", np.quantile(tg[(A[m] != ph[m]) & ok[m]], [0.1, .25, .5, .75, .9]))
