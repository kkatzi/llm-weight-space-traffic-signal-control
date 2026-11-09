"""Which hidden units can be repurposed with the least disturbance to the original decisions?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools

w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"]
ok = X[:, 44] > 0.5
Xo = X[ok]
g0 = tools.greedy(w, Xo)
h1, h2 = tools.hidden(w, Xo)

print("--- h1 units: flip rate when replaced by its mean ---")
res1 = []
for i in range(64):
    h1m = h1.copy(); h1m[:, i] = h1[:, i].mean()
    h2m = np.tanh(h1m @ w["W2"].T + w["b2"])
    g = (h2m @ w["W3"].T + w["b3"]).argmax(1)
    res1.append(((g != g0).mean(), i))
res1.sort()
print([(i, round(f, 4)) for f, i in res1[:12]])

print("--- h2 units: flip rate when replaced by its mean ---")
res2 = []
for j in range(64):
    h2m = h2.copy(); h2m[:, j] = h2[:, j].mean()
    g = (h2m @ w["W3"].T + w["b3"]).argmax(1)
    res2.append(((g != g0).mean(), j))
res2.sort()
print([(j, round(f, 4)) for f, j in res2[:12]])

# joint: best 3 h1 + 3 h2
best1 = [i for _, i in res1[:3]]
best2 = [j for _, j in res2[:3]]
h1m = h1.copy()
for i in best1:
    h1m[:, i] = h1[:, i].mean()
h2m = np.tanh(h1m @ w["W2"].T + w["b2"])
for j in best2:
    h2m[:, j] = h2[:, j].mean()
g = (h2m @ w["W3"].T + w["b3"]).argmax(1)
print("joint flip rate for h1", best1, "h2", best2, ":", (g != g0).mean())
