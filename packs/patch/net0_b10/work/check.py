"""Checks for an edited network: agreement with the original on recorded states, gate behaviour, deadlock probe."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
w0 = tools.load('start_weights.npz'); w = tools.load(sys.argv[1])
print('edit size', tools.edit_size(w, w0))
S = tools.states(); X = S['obs'].astype(np.float64); ctrl = S['controller']
N = tools.names(); ix = {n: i for i, n in enumerate(N)}
SL = {0: ['N0', 'N1', 'S0', 'S1'], 1: ['N2', 'S2'], 2: ['E0', 'E1', 'W0', 'W1'], 3: ['E2', 'W2']}
UP = {0: ['N0u', 'N1u', 'S0u', 'S1u'], 1: ['N1u', 'S1u'], 2: ['E0u', 'E1u', 'W0u', 'W1u'], 3: ['E1u', 'W1u']}
dem = np.stack([sum(X[:, ix['d_' + l]] for l in SL[p]) + sum(X[:, ix['d_' + u]] for u in UP[p]) for p in range(4)], 1)
a0 = tools.greedy(w0, X); a = tools.greedy(w, X)
ok = X[:, 44] > 0.5; ph = X[:, 40:44].argmax(1)
for c in range(3):
    m = (ctrl == c) & ok
    print('ctrl %d ok-states %d  agreement %.4f  (all states %.4f)' % (c, m.sum(), (a == a0)[m].mean(), (a == a0)[ctrl == c].mean()))
anyd = dem.sum(1) > 0
chose_empty0 = ok & anyd & (dem[np.arange(len(X)), a0] == 0)
chose_empty1 = ok & anyd & (dem[np.arange(len(X)), a] == 0)
print('ok-states where chosen phase has no demand but some phase does: original %d  edited %d' % (chose_empty0.sum(), chose_empty1.sum()))
dis = ok & (a != a0)
print('disagreements (ok):', dis.sum(), ' of which original chose empty phase:', (dis & chose_empty0).sum())
# breakdown of disagreements not explained by the gate
other = dis & ~chose_empty0
print('unexplained disagreements:', other.sum())
print(tools.trainability(w))
