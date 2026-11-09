"""Scan layer-1 units as candidates to repurpose into an 'all visible lanes empty' detector.
For each unit k: decision agreement when its output is pinned to a constant (-1 or +1) with the mean contribution
moved into b2 (what repurposing costs on recorded states), and whether flipping it in the empty states pushes the
network to phase 3 (EW_left) through its existing W2 column."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools

S = tools.states(); X = S["obs"].astype(float); c = S["controller"]; ep = S["episode"]
w0 = tools.load("start_weights.npz")
wi = {k: v.copy() for k, v in w0.items()}
wi["W1"][:, 5] += w0["W1"][:, 11]; wi["W1"][:, 17] += w0["W1"][:, 23]
ok = X[:, 44] > 0.5
h1, h2 = tools.hidden(wi, X)
g = tools.greedy(wi, X)
norm = ~((c == 0) & (ep >= 2))      # exclude the gridlocked start-network episodes for the mean
# empty states
E = []
for p in range(4):
    for kk in range(12):
        tg = min(1.0, (2 + 5 * kk) / 60)
        x = np.zeros(46); x[40 + p] = 1; x[45] = tg; x[44] = 1.0 if tg >= 10 / 60 else 0.0
        E.append(x)
E = np.array(E); okE = E[:, 44] > 0.5
hE, _ = tools.hidden(wi, E)

def logits_from_h1(H):
    return np.tanh(H @ wi["W2"].T + wi["b2"]) @ wi["W3"].T + wi["b3"]

rows = []
for k in range(64):
    m = h1[norm, k].mean()
    for off in (-1.0, 1.0):
        on = -off
        b2c = wi["b2"] + wi["W2"][:, k] * (m - off)
        H = h1.copy(); H[:, k] = off
        L = np.tanh(H @ wi["W2"].T + b2c) @ wi["W3"].T + wi["b3"]
        agree = (L.argmax(1)[ok] == g[ok]).mean()
        HE = hE.copy(); HE[:, k] = on
        LE = np.tanh(HE @ wi["W2"].T + b2c) @ wi["W3"].T + wi["b3"]
        marg = (LE[:, 3] - np.delete(LE, 3, axis=1).max(1))[okE]
        rows.append((agree, k, off, m, marg.min(), h1[norm, k].std()))
rows.sort(reverse=True)
for r in rows[:20]:
    print("agree %.4f unit %2d off %+.0f mean %+.3f  empty-state min margin for phase3 %+.2f  std %.3f" % r)
