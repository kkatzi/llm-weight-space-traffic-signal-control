import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w0 = tools.load('start_weights.npz')
S = tools.states(); X = S['obs'].astype(np.float64); C = S['controller']
ok = X[:,44] > 0.5
cur = X[:,40:44].argmax(1)
load = X[:, :40].sum(1)
# empty phase-2 state: which W2 value (true input to original net) makes phase 3 win?
for tg in [0.2, 0.5, 0.75, 1.0]:
    out = []
    for q in np.linspace(0, 1, 11):
        x = np.zeros(46); x[42] = 1; x[44] = 1; x[45] = tg; x[11] = q; x[23] = q
        out.append(tools.forward(w0, x)[0].argmax())
    print('empty ph2 tg', tg, 'argmax vs W2 (q=d) 0..1:', out)
    out = []
    for q in np.linspace(0, 1, 11):
        x = np.zeros(46); x[42] = 1; x[44] = 1; x[45] = tg; x[11] = q; x[23] = 0
        out.append(tools.forward(w0, x)[0].argmax())
    print('   only q set:', out)
# held phase-2 states in recorded data
m = ok & (cur == 2)
for lo, hi in [(0.2, 0.21), (0.25, 0.5), (0.5, 0.99), (0.99, 1.01)]:
    mm = m & (X[:,45] >= lo) & (X[:,45] < hi)
    print('ph2 tg in', (lo, hi), 'n', mm.sum(), 'empty share', (load[mm] < 1e-6).mean().round(3), 'median load', np.median(load[mm]).round(2) if mm.any() else None,
          'by ctrl', np.bincount(C[mm], minlength=3))
