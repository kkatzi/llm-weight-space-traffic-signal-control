"""analyse.py: compare decisions of candidate networks on recorded states (forward passes only)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import tools
from build import cand, W0, X, CTRL, PH, spill_states
np.set_printoptions(precision=3, suppress=True, linewidth=160)
ok = X[:, 44] > 0.5
g0 = tools.greedy(W0, X)
# references with a "true-ish" W2 reading: E2 copy, and full on spill states
Xsub = X.copy(); Xsub[:, 11] = X[:, 5]; Xsub[:, 23] = X[:, 17]
sp = spill_states()
Xfull = X.copy(); Xfull[sp, 11] = 1; Xfull[sp, 23] = 1
gfull = tools.greedy(W0, Xfull)
names = sys.argv[1:]
for nm in names:
    w = cand(nm)
    g = tools.greedy(w, X)
    print(f'== {nm}  changed={tools.edit_size(w, W0)["changed"]}')
    for ci in range(3):
        m = (CTRL == ci) & ok
        print(f'  ctrl{ci} act dist {np.bincount(g[m], minlength=4)/m.sum()}  agree-start {(g[m]==g0[m]).mean():.3f}')
    m = sp & ok
    print(f'  spill states n={m.sum()}: dist {np.bincount(g[m], minlength=4)/m.sum()}  agree with W0(W2=full) {(g[m]==gfull[m]).mean():.3f}  start {np.bincount(g0[m],minlength=4)/m.sum()}  W0(full) {np.bincount(gfull[m],minlength=4)/m.sum()}')
    m = (~sp) & ok
    print(f'  non-spill n={m.sum()}: dist {np.bincount(g[m], minlength=4)/m.sum()}')
    tr = tools.trainability(w)
    print('  ', {k: round(v, 3) for k, v in tr.items()})
