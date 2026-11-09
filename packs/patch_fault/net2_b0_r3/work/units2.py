import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools, numpy as np
from build import impute_w2
w0 = tools.load("start_weights.npz"); w = impute_w2(w0)
S = tools.states(); X = S["obs"].astype(np.float64)
h1 = np.tanh(X @ w["W1"].T + w["b1"])
a2 = h1 @ w["W2"].T + w["b2"]; h2 = np.tanh(a2)
base = tools.greedy(w, X)
res = []
for u in range(64):
    out = []
    for v in [-1.0, 1.0]:
        h1c = h1.copy(); h1c[:, u] = v
        lg = np.tanh(h1c @ w["W2"].T + w["b2"]) @ w["W3"].T + w["b3"]
        out.append(np.mean(lg.argmax(1) != base))
    res.append((u, *out, np.abs(w["W2"][:, u]).sum()))
print("layer-1 units best when clamped to -1 / +1:")
for r in sorted(res, key=lambda r: min(r[1], r[2]))[:10]:
    print("  u=%2d imp(-1) %.4f imp(+1) %.4f  sum|W2[:,u]| %.2f" % r)
print("layer-2 units: clamp to mean, W3 col norm, pre range")
res = []
for j in range(64):
    h2c = h2.copy(); h2c[:, j] = h2[:, j].mean()
    lg = h2c @ w["W3"].T + w["b3"]
    res.append((j, np.mean(lg.argmax(1) != base), np.linalg.norm(w["W3"][:, j]), a2[:, j].min(), a2[:, j].max()))
for r in sorted(res, key=lambda r: r[1])[:8]:
    print("  j=%2d imp %.4f |W3col| %.3f pre[%.2f, %.2f]" % r)
