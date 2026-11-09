import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
obs, ctl = S['obs'].astype(np.float64), S['controller']
ok = obs[:, 44] > .5
X = obs[ok]
g0 = tools.greedy(w, X)
h1, h2 = tools.hidden(w, X)
res1 = []
for k in range(64):
    w2 = {kk: v.copy() for kk, v in w.items()}
    m = h1[:, k].mean()
    # replace unit k by its mean: b2 += W2[:,k]*m, W2[:,k]=0
    w2["b2"] = w2["b2"] + w2["W2"][:, k]*m; w2["W2"][:, k] = 0
    res1.append(((tools.greedy(w2, X) != g0).mean(), h1[:, k].std(), np.abs(w["W2"][:, k]).sum()))
res1 = np.array(res1)
o = np.argsort(res1[:, 0])
print("h1 ablation (flip share, std, |W2 col| sum), least important first:")
for k in o[:12]: print(k, res1[k])
res2 = []
for j in range(64):
    w2 = {kk: v.copy() for kk, v in w.items()}
    m = h2[:, j].mean()
    w2["b3"] = w2["b3"] + w2["W3"][:, j]*m; w2["W3"][:, j] = 0
    res2.append(((tools.greedy(w2, X) != g0).mean(), h2[:, j].std(), np.abs(w["W3"][:, j]).sum()))
res2 = np.array(res2)
o2 = np.argsort(res2[:, 0])
print("h2 ablation (flip share, std, |W3 col| sum), least important first:")
for j in o2[:12]: print(j, res2[j])
