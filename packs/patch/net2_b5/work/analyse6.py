import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load('start_weights.npz')
S = tools.states()
X = S['obs'].astype(float); A = S['act']; E = S['episode']; T = S['time_s']; C = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
UPS = {0: [24, 25, 28, 29], 1: [25, 29], 2: [26, 27, 30, 31], 3: [27, 31]}
g = tools.greedy(w, X)
UQ = np.stack([X[:, UPS[p]].sum(1) * 34.13 for p in range(4)], 1)
q = X[:, :12] * 3
Q = np.stack([q[:, SERVE[p]].sum(1) for p in range(4)], 1)
# states with significant upstream queues, ok=1, from all controllers
m = ok & (X[:, 24:32].max(1) * 34.13 >= 4)
idx = np.where(m)[0]
print('n', len(idx))
cnt = 0
for i in idx[::max(1, len(idx)//40)]:
    print(C[i], T[i], 'cur', ph[i], 'tg', round(X[i,45]*60), 'net', g[i], 'Q', Q[i].round(0), 'UQ', UQ[i].round(0), 'uq', (X[i,24:32]*34.13).round(0), 'q', q[i].round(0))
