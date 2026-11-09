import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load('start_weights.npz')
S = tools.states()
X = S['obs'].astype(float); E = S['episode']; T = S['time_s']; C = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
UPS = {0: [24, 25, 28, 29], 1: [25, 29], 2: [26, 27, 30, 31], 3: [27, 31]}
g = tools.greedy(w, X)
d = X[:, 12:24] * 3; q = X[:, :12] * 3
D = np.stack([d[:, SERVE[p]].sum(1) for p in range(4)], 1)
Q = np.stack([q[:, SERVE[p]].sum(1) for p in range(4)], 1)
UQ = np.stack([X[:, UPS[p]].sum(1) * 34.13 for p in range(4)], 1)
UD = np.stack([X[:, [i + 8 for i in UPS[p]]].sum(1) * 34.13 for p in range(4)], 1)
n = np.arange(len(X))
m = (C == 0) & ok
cur = ph
others_q = Q.sum(1) - Q[n, cur] + UQ.sum(1) - UQ[n, cur]
H, xe, ye = np.histogram2d(np.clip(others_q[m], 0, 20), np.clip(UD[m, cur[m]] + D[m, cur[m]], 0, 15), bins=[[0, 1, 2, 3, 5, 8, 12, 21], [0, 1, 2, 4, 6, 9, 16]])
print('rows: others halting bins', xe, 'cols: current approaching', ye)
print(H.astype(int))
# which time of episode -> demand interval
for lo, hi in [(0, 900), (900, 1800), (1800, 2700), (2700, 3600), (3600, 6000)]:
    mm = m & (T >= lo) & (T < hi)
    print(lo, hi, 'ok', mm.sum(), 'mean others halting', others_q[mm].mean().round(2), 'total halting', (Q.sum(1) + UQ.sum(1)/2)[mm].mean().round(2))
