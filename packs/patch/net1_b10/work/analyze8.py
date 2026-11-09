import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); A = S['act']; ep = S['episode']; t = S['time_s']; c = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
capS = 22.4/7.5; capU = 256/7.5
PL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
PU = {0: [0,1,4,5], 1: [1,5], 2: [2,3,6,7], 3: [3,7]}
q = X[:, 0:12]*capS; d = X[:, 12:24]*capS; qu = X[:, 24:32]*capU; du = X[:, 32:40]*capU
L = tools.forward(w, X); g = L.argmax(1)
N = len(X)
Q = np.stack([q[:, PL[p]].sum(1) for p in range(4)], 1)   # halting at stop line per phase
D = np.stack([d[:, PL[p]].sum(1) for p in range(4)], 1)
np.set_printoptions(precision=2, suppress=True, linewidth=160)
for ci in range(3):
    m = ok & (c == ci)
    sw = m & (g != ph)
    Qg = Q[np.arange(N), g]
    Qcur = Q[np.arange(N), ph]; Dcur = D[np.arange(N), ph]
    others = Q.sum(1) - Qcur
    print(S['controller_names'][ci], 'ok states', m.sum())
    print('  net switches to phase with 0 halting:', (sw & (Qg < 0.5)).sum(), ' of switches', sw.sum())
    print('  ok states where all other phases have 0 halting:', (m & (others < 0.5)).sum(), ' net holds there', (m & (others<0.5) & (g==ph)).sum())
    print('  ok states with others halting<=1:', (m & (others < 1.5)).sum(), ' net holds there', (m & (others<1.5) & (g==ph)).sum())
    # distribution of total other halting at ok
    print('  other-halting pct 10/25/50/75/90', np.percentile(others[m], [10,25,50,75,90]))
    print('  current-phase stop-line density (moving) pct 50/75/90', np.percentile(Dcur[m], [50,75,90]))
