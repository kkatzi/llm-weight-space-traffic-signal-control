import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); A = S['act']; ep = S['episode']; t = S['time_s']; c = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
capS = 22.4/7.5; capU = 256/7.5
PL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
q = X[:, 0:12]*capS; d = X[:, 12:24]*capS; qu = X[:, 24:32]*capU; du = X[:, 32:40]*capU
L = tools.forward(w, X); g = L.argmax(1); N = len(X)
Q = np.stack([q[:, PL[p]].sum(1) for p in range(4)], 1)
D = np.stack([d[:, PL[p]].sum(1) for p in range(4)], 1)
np.set_printoptions(precision=2, suppress=True, linewidth=160)
m = ok & (g != ph)
Qx = Q.copy(); Qx[np.arange(N), ph] = -9
best = Qx.argmax(1)
C = np.zeros((4,4), int)
for a, b in zip(best[m], g[m]): C[a, b] += 1
print('rows: max-halting phase, cols: net pick (ok switching states, all controllers)\n', C)
# mean Q per phase in disagreement cells
for a in range(4):
    for b in range(4):
        mm = m & (best == a) & (g == b)
        if a != b and mm.sum() >= 30:
            print(f'maxQ={a} net={b} n={mm.sum()}: mean Q', Q[mm].mean(0), ' D', D[mm].mean(0), ' cur phase hist', np.bincount(ph[mm], minlength=4))
# per lane mean halting when net picks each phase
for b in range(4):
    mm = m & (g == b)
    print(f'net picks {b}: mean Q per phase', Q[mm].mean(0), ' per-lane q', q[mm].mean(0))
