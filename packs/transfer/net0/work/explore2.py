import numpy as np
from collections import Counter
from common import *
np.set_printoptions(precision=3, suppress=True, linewidth=200)
L2 = tools.states('states_L2.npz'); X2 = L2['obs'].astype(float)
S = tools.states(); X = S['obs'].astype(float)

def greens(X, ep, tg_idx, ph_sl):
    """Green episodes: (phase, max t_green*60 seen)"""
    out = []; trans = Counter()
    for e in np.unique(ep):
        m = ep == e
        ph = X[m][:, ph_sl].argmax(1); tg = X[m][:, tg_idx] * 60
        start = 0
        for i in range(1, len(ph) + 1):
            if i == len(ph) or ph[i] != ph[i - 1]:
                out.append((ph[start], tg[start:i].max(), i - start))
                if i < len(ph): trans[(ph[i - 1], ph[i])] += 1
                start = i
    return out, trans

g, tr = greens(X2, L2['episode'], 45, slice(40, 44))
print('L2 net on L2: greens per phase: count, mean max t_green, quantiles')
for p in range(4):
    d = np.array([x[1] for x in g if x[0] == p])
    print(p, len(d), d.mean().round(1), np.quantile(d, [.1, .5, .9]).round(1))
print('transitions', sorted(tr.items()))
print('episode lengths (decisions):', [int((L2['episode'] == e).sum()) for e in np.unique(L2['episode'])])
for c in [0, 1]:
    m = S['controller'] == c
    g, tr = greens(X[m], S['episode'][m], 53, slice(48, 52))
    print('L2W ctrl', c)
    for p in range(4):
        d = np.array([x[1] for x in g if x[0] == p])
        print(p, len(d), d.mean().round(1), np.quantile(d, [.1, .5, .9]).round(1))
    print('episode lengths:', [int(((S['episode'] == e) & m).sum()) for e in np.unique(S['episode'][m])])
