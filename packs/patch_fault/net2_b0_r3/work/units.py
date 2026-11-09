import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools, numpy as np
from build import impute_w2
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w0 = tools.load("start_weights.npz"); w = impute_w2(w0)
S = tools.states(); X = S["obs"].astype(np.float64)
# add synthetic sparse states (lone cars, empty) across phases and green times
syn = []
for lane in [None] + list(range(11)):
    for ncar in [1, 2]:
        for k in range(4):
            for g in [2, 7, 12, 22, 32, 47, 62]:
                x = np.zeros(46)
                if lane is not None:
                    v = ncar * 0.335; x[lane] = v; x[12 + lane] = v
                x[40 + k] = 1; x[44] = 1.0 if g >= 10 else 0; x[45] = min(g / 60, 1)
                syn.append(x)
XA = np.vstack([X, np.array(syn)])
a1 = XA @ w["W1"].T + w["b1"]; h1 = np.tanh(a1)
a2 = h1 @ w["W2"].T + w["b2"]; h2 = np.tanh(a2)
base = tools.greedy(w, X)
n = len(X)
print("layer-1 units: pre-activation min/max over recorded+synthetic, h std, importance (decision change when clamped to mean)")
rows = []
for u in range(64):
    h1c = h1[:n].copy(); h1c[:, u] = h1[:n, u].mean()
    lg = np.tanh(h1c @ w["W2"].T + w["b2"]) @ w["W3"].T + w["b3"]
    imp = np.mean(lg.argmax(1) != base)
    rows.append((u, a1[:, u].min(), a1[:, u].max(), h1[:, u].std(), imp))
for r in sorted(rows, key=lambda r: r[4])[:12]:
    print("  u=%2d pre[%.2f, %.2f] std %.3f imp %.4f" % r)
print("layer-2 units:")
rows = []
for j in range(64):
    h2c = h2[:n].copy(); h2c[:, j] = h2[:n, j].mean()
    lg = h2c @ w["W3"].T + w["b3"]
    imp = np.mean(lg.argmax(1) != base)
    rows.append((j, a2[:, j].min(), a2[:, j].max(), h2[:, j].std(), imp, *w["W3"][:, j]))
for r in sorted(rows, key=lambda r: r[4])[:12]:
    print("  j=%2d pre[%.2f, %.2f] std %.3f imp %.4f  W3col [%.2f %.2f %.2f %.2f]" % r)
