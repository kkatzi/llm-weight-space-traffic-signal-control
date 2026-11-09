import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load('start_weights.npz')
S = tools.states()
X = S['obs'].astype(float); E = S['episode']; T = S['time_s']; C = S['controller']
names = tools.names()
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
g = tools.greedy(w, X)
d = X[:, 12:24]
D = np.stack([d[:, SERVE[p]].sum(1) for p in range(4)], 1)
m3 = ok & (ph == 3) & (D[:, 3] == 0)
bad = m3 & (g == 3); good = m3 & (g != 3)
print('bad', bad.sum(), 'good', good.sum())
diff = X[bad].mean(0) - X[good].mean(0)
order = np.argsort(-np.abs(diff))
for j in order[:16]:
    print(names[j], 'bad mean', X[bad, j].mean().round(3), 'good mean', X[good, j].mean().round(3))
print('t_green bad', np.bincount(np.round(X[bad, 45] * 60).astype(int))[10:40:5], 'good', np.bincount(np.round(X[good, 45] * 60).astype(int))[10:40:5])
# what do 'good' choose
print('good picks', np.bincount(g[good], minlength=4))
# same for other ok states where current phase is empty and it's phase 2/0
