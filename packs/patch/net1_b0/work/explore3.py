import sys; sys.path.insert(0, '.')
import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'], S['episode'], S['time_s'], S['controller']
N = tools.names(); ix = {n: i for i, n in enumerate(N)}
g = tools.greedy(w, obs)
ph = obs[:, 40:44].argmax(1)
ok = obs[:, 44] > 0.5
for c in range(3):
    m = (ctl == c)
    print("ctl", c, "ok share", ok[m].mean(), "net picks current when ok", (g[m & ok] == ph[m & ok]).mean(), "when not ok", (g[m & ~ok] == ph[m & ~ok]).mean())
    print("  net pick distribution (ok):", np.bincount(g[m & ok], minlength=4) / (m & ok).sum())
    print("  tgreen values ok:", np.unique(np.round(obs[m & ok, 45]*60)))
# what green ages appear
m = ctl == 0
print("tgreen*60 for net episodes:", np.unique(np.round(obs[m, 45]*60), return_counts=True))
print("ok vs tgreen:", [(a, obs[m & (np.round(obs[:,45]*60)==a), 44].mean()) for a in [2,7,12,17,22]])
