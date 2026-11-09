"""Pick h1 units to turn into trigger units (forced to -1 in normal operation) and an h2 relay unit."""
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
import tools
np.set_printoptions(precision=3, suppress=True, linewidth=220)
S = tools.states(); X = S["obs"].astype(float); c = S["controller"]; ep = S["episode"]
w = tools.load(os.path.join(HERE, "work", "v1.npz"))
m = (c > 0) | (ep < 2)
X = X[m]; ok = X[:, 44] > 0.5
h1, h2 = tools.hidden(w, X)
base = tools.greedy(w, X)
def lg_h1(H1, b2=None):
    return np.tanh(H1 @ w["W2"].T + (w["b2"] if b2 is None else b2)) @ w["W3"].T + w["b3"]
res = []
for j in range(64):
    H = h1.copy(); H[:, j] = -1.0
    f_nocomp = ((lg_h1(H).argmax(1) != base)[ok]).mean()
    b2c = w["b2"] + w["W2"][:, j] * (h1[:, j].mean() + 1.0)
    f_comp = ((lg_h1(H, b2c).argmax(1) != base)[ok]).mean()
    res.append((f_nocomp, f_comp, j))
res.sort()
for f, fc, j in res[:12]:
    print(f"j={j:2d} flips(h1_j=-1, no comp)={f:.4f}  with b2 comp={fc:.4f}  |W2[:,j]|={np.linalg.norm(w['W2'][:, j]):.2f}")
# quantisation of a few inputs
for i in [31, 22, 27, 16]:
    v = np.unique(np.round(X[:, i], 4)); print(tools.names()[i], v[:8])
