import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load('start_weights.npz')
S = tools.states()
X = S['obs'].astype(float); A = S['act']; E = S['episode']; T = S['time_s']; C = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
g = tools.greedy(w, X)
q = X[:, :12]; d = X[:, 12:24]
Q = np.stack([q[:, SERVE[p]].sum(1) for p in range(4)], 1)
D = np.stack([d[:, SERVE[p]].sum(1) for p in range(4)], 1)
for c in range(3):
    m = (C == c) & ok
    idx = np.where(m)[0]
    pick = g[idx]
    empty_pick = (D[idx, pick] == 0)
    other_has_q = np.array([any(Q[i, p] > 0 for p in range(4) if p != pick[k]) for k, i in enumerate(idx)])
    print(S['controller_names'][c], 'ok', len(idx), 'pick empty(density) phase', empty_pick.sum(), 'of which others queued', (empty_pick & other_has_q).sum())
    ext = pick == ph[idx]
    print('   extends', ext.sum(), 'extends with current density 0', (ext & (D[idx, ph[idx]] == 0)).sum(), 'and others queued', (ext & (D[idx, ph[idx]] == 0) & other_has_q).sum())
    # all-empty situations
    allE = D[idx].sum(1) == 0
    print('   all stop-line empty', allE.sum(), 'net picks current in those', (ext & allE).sum(), 'pick dist', np.bincount(pick[allE], minlength=4))
m = (C == 0) & ok
idx = np.where(m)[0]
for i in idx:
    if D[i, g[i]] == 0 and Q[i].sum() > 0:
        print(T[i], 'cur', ph[i], 'pick', g[i], 'Q', (Q[i]*3).round(0), 'D', (D[i]*3).round(0), 'ud', (X[i,32:40]*34).round(0), 'logits', tools.forward(w, X[i])[0].round(1))
