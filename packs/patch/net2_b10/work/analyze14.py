import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]
ok = X[:, 44] > 0.5
ph = X[:, 40:44].argmax(1)
names = tools.names(); ix = {n: i for i, n in enumerate(names)}
# honoured states where S2 pocket full and not currently phase 1
base = X[ok & (X[:, ix["q_S2"]] > 0.99) & (ph != 1)]
print("n base", len(base))
for extra in [0, 1, 2, 3, 4, 6, 8]:
    Xs = base.copy()
    # treat extra upstream vehicles as extra pocket vehicles (q and d)
    Xs[:, ix["q_S2"]] = 1 + extra/ (22.4/7.5)
    Xs[:, ix["d_S2"]] = 1 + extra/ (22.4/7.5)
    lg = tools.forward(w, Xs)
    g = lg.argmax(1)
    print(f"extra {extra}: picks phase1 share {(g==1).mean():.2f}; mean logit1 - max(others) {(lg[:,1] - np.delete(lg, 1, 1).max(1)).mean():.2f}")
base = X[ok & (X[:, ix["q_E2"]] > 0.99) & (ph != 3)]
print("n base E2", len(base))
for extra in [0, 1, 2, 3, 4, 6, 8]:
    Xs = base.copy()
    Xs[:, ix["q_E2"]] = 1 + extra/ (22.4/7.5)
    Xs[:, ix["d_E2"]] = 1 + extra/ (22.4/7.5)
    lg = tools.forward(w, Xs)
    g = lg.argmax(1)
    print(f"extra {extra}: picks phase3 share {(g==3).mean():.2f}; mean logit3 - max(others) {(lg[:,3] - np.delete(lg, 3, 1).max(1)).mean():.2f}")
# through lane 0 extrapolation
for lane, p in [("N0", 0), ("E0", 2), ("S1", 0), ("W1", 2)]:
    base = X[ok & (X[:, ix["q_"+lane]] > 0.99) & (ph != p)]
    out = []
    for extra in [0, 2, 4, 8]:
        Xs = base.copy()
        Xs[:, ix["q_"+lane]] = 1 + extra/(22.4/7.5); Xs[:, ix["d_"+lane]] = 1 + extra/(22.4/7.5)
        lg = tools.forward(w, Xs); out.append(round((lg.argmax(1) == p).mean(), 2))
    print(lane, "n", len(base), "share picks", p, "for extra 0,2,4,8:", out)
