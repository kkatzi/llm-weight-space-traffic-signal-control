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
q = X[:, :12] * 3; d = X[:, 12:24] * 3
uq = X[:, 24:32] * 34.13; ud = X[:, 32:40] * 34.13
Q = np.stack([q[:, SERVE[p]].sum(1) for p in range(4)], 1)
D = np.stack([d[:, SERVE[p]].sum(1) for p in range(4)], 1)
UQ = np.stack([X[:, UPS[p]].sum(1) * 34.13 for p in range(4)], 1)
UD = np.stack([X[:, [i + 8 for i in UPS[p]]].sum(1) * 34.13 for p in range(4)], 1)
m = (C == 0) & ok & (g != ph)
idx = np.where(m)[0]
cur = ph[idx]
print('at switch: current phase stop-line vehicles (D) dist', np.bincount(np.round(D[idx, cur]).astype(int)))
print('at switch: current phase upstream density veh', np.quantile(UD[idx, cur], [.5, .75, .9, .99]).round(1))
print('at switch: current phase upstream queue veh', np.quantile(UQ[idx, cur], [.5, .75, .9, .99]).round(1))
# queue (halting) totals at switch on other phases
oth = Q[idx].sum(1) - Q[idx, cur]
print('others halting', np.quantile(oth, [.1, .5, .9]).round(1))
# upstream queues: when they occur, in which lanes, which controller
for c in range(3):
    mm = C == c
    print(S['controller_names'][c], 'mean upstream queue per lane veh', (X[mm, 24:32] * 34.13).mean(0).round(2), ' max', (X[mm, 24:32] * 34.13).max(0).round(0))
    print('    mean stopline queue per lane veh', (X[mm, 0:12] * 3).mean(0).round(2))
