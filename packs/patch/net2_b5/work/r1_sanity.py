import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w0 = tools.load('start_weights.npz'); w = tools.load(sys.argv[1])
S = tools.states()
X = S['obs'].astype(float); C = S['controller']; T = S['time_s']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
g0 = tools.greedy(w0, X); g = tools.greedy(w, X)
q = np.round(X[:, :12] * 3); d = np.round(X[:, 12:24] * 3)
Q = np.stack([q[:, SERVE[p]].sum(1) for p in range(4)], 1)
D = np.stack([d[:, SERVE[p]].sum(1) for p in range(4)], 1)
idx = np.where(ok & (g0 == 3) & (ph == 3) & (g != 3))[0]
for i in idx:
    print(C[i], T[i], 'tg', round(X[i, 45]*60), 'new', g[i], 'Q', Q[i], 'D', D[i], 'ud', (X[i, 32:40]*34.13).round(0))
print('new pick has halting vehicles:', (Q[idx, g[idx]] > 0).mean(), ' is max-Q phase:', (Q[idx, g[idx]] == Q[idx].max(1)).mean())
