import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
from make_ramp import build
np.set_printoptions(precision=3, suppress=True, linewidth=200)
S = tools.states(); X = S['obs'].astype(np.float64); C = S['controller']
ok = X[:,44] > 0.5; cur = X[:,40:44].argmax(1)
wp = build(0.0)
gp = tools.greedy(wp, X)
for gamma in [0.75, 1.0, 1.5, 2.0]:
    w = build(gamma)
    g = tools.greedy(w, X)
    # equivalence check: network on failed obs == original on imputed obs
    w0 = tools.load('start_weights.npz')
    Xi = X.copy(); r = gamma * (X[:,45] - 0.2); Xi[:,11] = X[:,5] + r; Xi[:,23] = X[:,17] + r
    assert np.allclose(tools.forward(w, X), tools.forward(w0, Xi), atol=1e-9)
    trig = []
    for ph in range(4):
        row = []
        for tgs in range(12, 65, 5):
            x = np.zeros(46); x[40+ph] = 1; x[44] = 1; x[45] = min(tgs/60, 1)
            row.append(int(tools.forward(w, x)[0].argmax()))
        trig.append(row)
    m0 = ok & (C == 0)
    print(f'gamma {gamma}: agree w/ proxy (net states, ok) {(g[m0]==gp[m0]).mean():.4f}, all ok {(g[ok]==gp[ok]).mean():.4f}')
    print('   empty-state choice per phase at t_green 12,17,...,62 s:')
    for ph in range(4): print('    ph', ph, trig[ph])
    print('   changed decisions by current phase (net states):', [int(((g!=gp)&m0&(cur==p)).sum()) for p in range(4)])
