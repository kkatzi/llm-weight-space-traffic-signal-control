import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=3, suppress=True)
w = tools.load('start_weights.npz')
S = tools.states()
X = S['obs'].astype(float); A = S['act']; E = S['episode']; T = S['time_s']; C = S['controller']
ph = X[:, 40:44].argmax(1)
ok = X[:, 44] > 0.5
tg = np.round(X[:, 45] * 60)
# lanes served by each phase (stop-line indices)
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
UP = {0: [24, 25, 28, 29], 1: [25, 29], 2: [26, 27, 30, 31], 3: [27, 31]}
def phase_q(x):
    return np.stack([x[:, SERVE[p]].sum(1) for p in range(4)], 1)
def phase_d(x):
    return np.stack([x[:, [i + 12 for i in SERVE[p]]].sum(1) for p in range(4)], 1)
for c, name in enumerate(S['controller_names']):
    m = C == c
    # green durations: count consecutive decisions in same phase per episode
    durs = []
    for e in np.unique(E[m]):
        idx = np.where(m & (E == e))[0]
        p = ph[idx]
        tt = tg[idx]
        # a green ends when the next decision's t_green is 2 (new green)
        for k in range(len(idx) - 1):
            if tt[k + 1] == 2 and k > 0:
                durs.append(tt[k] + 3)  # approx: decided at tt[k] then + yellow
    durs = np.array(durs)
    print(name, 'n greens', len(durs), 'mean len', durs.mean().round(1), 'quantiles', np.quantile(durs, [.1, .25, .5, .75, .9]))
g = tools.greedy(w, X)
m = (C == 0) & ok
q = phase_q(X); d = phase_d(X)
print('start net, ok states', m.sum(), 'switch share', (g[m] != ph[m]).mean())
# when switching, to a phase with zero queue AND zero density?
sw = m & (g != ph)
tgt = g[sw]
print('switch to phase with empty queue', (q[sw, tgt] == 0).mean(), 'empty density', (d[sw, tgt] == 0).mean())
# when it switches, current phase queue
print('switch away while current phase has queue>0:', (q[sw, ph[sw]] > 0).mean())
# best alternative?
for p in range(4):
    mm = m & (ph == p)
    print('phase', p, 'ok decisions', mm.sum(), 'pick dist', np.bincount(g[mm], minlength=4))
