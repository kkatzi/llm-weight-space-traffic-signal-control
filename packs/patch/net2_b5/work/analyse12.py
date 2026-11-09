import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load('start_weights.npz')
S = tools.states()
X = S['obs'].astype(float); E = S['episode']; T = S['time_s']; C = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
g = tools.greedy(w, X)
q = np.round(X[:, :12] * 3).astype(int)
m = (C == 0) & ok & (g != ph)
for p in range(4):
    mm = m & (g == p)
    tot = q[mm][:, SERVE[p]].sum(1)
    mx = q[mm][:, SERVE[p]].max(1)
    print('switch to', p, 'n', mm.sum(), 'queued veh on its lanes dist', np.bincount(tot, minlength=13)[:13], ' max-lane-queue dist', np.bincount(mx, minlength=4))
# for comparison, at the same decisions, the queue of phases not picked
for p in range(4):
    mm = m & (g != p) & (ph != p)
    tot = q[mm][:, SERVE[p]].sum(1)
    print('not picked', p, 'n', mm.sum(), 'queued dist', np.bincount(tot, minlength=13)[:13])
