import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
obs, ctl = S['obs'].astype(np.float64), S['controller']
ok = obs[:, 44] > .5
h1, h2 = tools.hidden(w, obs[ok])
X = obs[ok]; g0 = tools.greedy(w, X)
print("h1 units: idx, mean, std, |mean|*|W2col|sum")
rows = []
for k in range(64):
    w2 = {kk: v.copy() for kk, v in w.items()}
    w2["W2"][:, k] = 0          # ablate to zero (no compensation)
    flip0 = (tools.greedy(w2, X) != g0).mean()
    rows.append((k, h1[:, k].mean(), h1[:, k].std(), flip0))
rows = sorted(rows, key=lambda r: r[3])
for r in rows[:10]: print("  h1", r)
rows2 = []
for j in range(64):
    w2 = {kk: v.copy() for kk, v in w.items()}
    w2["b3"] = w2["b3"] + w2["W3"][:, j]*h2[:, j].mean(); w2["W3"][:, j] = 0
    rows2.append((j, h2[:, j].mean(), h2[:, j].std(), (tools.greedy(w2, X) != g0).mean()))
rows2 = sorted(rows2, key=lambda r: r[3])
for r in rows2[:10]: print("  h2", r)
