"""Probe 2: random sparse 'drain-time' states (few vehicles, incl. upstream moving/halted ones).
Counts states where the chosen phase cannot serve any vehicle while some other phase can.
A vehicle on a stop-line lane is servable by its phase; a moving vehicle on an upstream lane is servable by the
phases its lane feeds; a halted upstream vehicle is only blocked behind its stop-line lane (which we fill)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
rng = np.random.default_rng(0)
N = tools.names(); ix = {n: i for i, n in enumerate(N)}
SL = {0: ['N0', 'N1', 'S0', 'S1'], 1: ['N2', 'S2'], 2: ['E0', 'E1', 'W0', 'W1'], 3: ['E2', 'W2']}
UP = {0: ['N0u', 'N1u', 'S0u', 'S1u'], 1: ['N1u', 'S1u'], 2: ['E0u', 'E1u', 'W0u', 'W1u'], 3: ['E1u', 'W1u']}
lanes = ['N0','N1','N2','E0','E1','E2','S0','S1','S2','W0','W1','W2']
ups = ['N0u','N1u','E0u','E1u','S0u','S1u','W0u','W1u']
vs, vu = 1/(22.4/7.5), 1/(256/7.5)
def sample(n):
    Xs, serv = [], []
    for _ in range(n):
        x = np.zeros(46); p = rng.integers(4); x[40+p] = 1; x[44] = 1; x[45] = rng.choice([12, 17, 30, 60])/60
        s = np.zeros(4, bool)
        for l in rng.choice(lanes, rng.integers(0, 4), replace=False):
            k = rng.integers(1, 4); x[ix['q_'+l]] = min(1, k*vs) * (rng.random() < 0.8); x[ix['d_'+l]] = min(1, k*vs)
            for q in range(4):
                if l in SL[q]: s[q] = True
        for u in rng.choice(ups, rng.integers(0, 3), replace=False):
            k = rng.integers(1, 4); x[ix['d_'+u]] = k*vu
            if rng.random() < 0.5:   # halted (blocked): its downstream lane must be full -> fill one fed lane
                x[ix['q_'+u]] = k*vu
                fed = [l for l in lanes if l[0] == u[0] and ((u[1] == '0' and l[1] == '0') or (u[1] == '1' and l[1] in '12'))]
                l = rng.choice(fed); x[ix['q_'+l]] = 1; x[ix['d_'+l]] = 1
                for q in range(4):
                    if l in SL[q]: s[q] = True
            else:
                for q in range(4):
                    if u in UP[q]: s[q] = True
        Xs.append(x); serv.append(s)
    return np.array(Xs), np.array(serv)
X, serv = sample(40000)
ph = X[:, 40:44].argmax(1)
for f in ['start_weights.npz'] + sys.argv[1:]:
    w = tools.load(f); a = tools.greedy(w, X)
    bad = serv.any(1) & ~serv[np.arange(len(X)), a]
    hold = bad & (a == ph)
    print('%-22s chosen phase has nothing to serve while another does: %5d / %d  (holding current: %d)' % (os.path.basename(f), bad.sum(), serv.any(1).sum(), hold.sum()))
# breakdown for the original network: which phase was wrongly chosen, and was it the current one
w = tools.load('start_weights.npz'); a = tools.greedy(w, X)
bad = serv.any(1) & ~serv[np.arange(len(X)), a]
print('original: wrongly chosen phase counts', np.bincount(a[bad], minlength=4), ' of which holding current', np.bincount(a[bad & (a == ph)], minlength=4))
