import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "work"))
import numpy as np, tools
w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
w = tools.load(os.path.join(HERE, "work", "cand_E.npz"))
print("edit size:", tools.edit_size(w, w0))
changed = {k: int((w[k] != w0[k]).sum()) for k in w}
print("changed per tensor:", changed)
names = tools.names()
cols = sorted(set(np.where((w["W1"] != w0["W1"]).any(0))[0]))
print("W1 columns touched:", [names[c] for c in cols], "entries per column:", [int((w["W1"][:, c] != w0["W1"][:, c]).sum()) for c in cols])
S = tools.states(); X = S["obs"].astype(float); ok = X[:, 44] > 0.5
print("trainability orig:", tools.trainability(w0))
print("trainability new :", tools.trainability(w))
# identical where no upstream queue
noq = (X[:, 24:32] == 0).all(1)
print("max |logit diff| where q_up==0:", np.abs(tools.forward(w, X[noq]) - tools.forward(w0, X[noq])).max())
# extreme synthetic: random recorded honoured states with heavy spill on one approach's lane-1 upstream + full pocket
rng = np.random.default_rng(0)
ix = {n: i for i, n in enumerate(names)}
left = {"N": 1, "S": 1, "E": 3, "W": 3}; thr = {"N": 0, "S": 0, "E": 2, "W": 2}
base = X[ok][rng.choice(ok.sum(), 400, replace=False)]
for a in "NESW":
    for n in [1, 3, 8, 20, 34]:
        Xs = base.copy(); Xs[:, ix[f"q_{a}2"]] = 1; Xs[:, ix[f"d_{a}2"]] = 1
        Xs[:, ix[f"q_{a}1u"]] = n / (256/7.5); Xs[:, ix[f"d_{a}1u"]] = max(Xs[:, ix[f"d_{a}1u"]].max(), n / (256/7.5))
        cur = Xs[:, 40:44].argmax(1)
        lg = tools.forward(w, Xs); g = lg.argmax(1); g0 = tools.forward(w0, Xs).argmax(1)
        m = cur != left[a]
        print(f"  {a} pocket full + {n:2d} halted on {a}1u: picks left {np.mean(g[m]==left[a]):.2f} (orig {np.mean(g0[m]==left[a]):.2f}); max|logit| {np.abs(lg).max():.1f}")
    for n in [3, 20]:
        Xs = base.copy(); Xs[:, ix[f"q_{a}0"]] = 1; Xs[:, ix[f"d_{a}0"]] = 1
        Xs[:, ix[f"q_{a}0u"]] = n / (256/7.5); Xs[:, ix[f"d_{a}0u"]] = np.maximum(Xs[:, ix[f"d_{a}0u"]], n / (256/7.5))
        cur = Xs[:, 40:44].argmax(1); g = tools.greedy(w, Xs); g0 = tools.greedy(w0, Xs); m = cur != thr[a]
        print(f"  {a} lane0 full + {n:2d} halted on {a}0u: picks through {np.mean(g[m]==thr[a]):.2f} (orig {np.mean(g0[m]==thr[a]):.2f})")
# competing spillbacks: N left spill 4 vs E left spill 4 -> choice split
Xs = base.copy()
for a in "NE":
    Xs[:, ix[f"q_{a}2"]] = 1; Xs[:, ix[f"d_{a}2"]] = 1; Xs[:, ix[f"q_{a}1u"]] = 4/(256/7.5)
g = tools.greedy(w, Xs); cur = Xs[:, 40:44].argmax(1)
print("competing N/E left spill (4 each), not in 1/3:", np.bincount(g[(cur != 1) & (cur != 3)], minlength=4))
