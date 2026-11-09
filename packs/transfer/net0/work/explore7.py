# Exploration script (run during design). "plain()" = lane mapping without the unstick term.
"""Unstick term: x_L2[q,d of chosen lanes] += kappa * (t_green/60 - 0.2). Exactly inert at t_green = 12 s."""
import numpy as np, itertools
from common import *
from explore6 import x_w, SERVE
np.set_printoptions(precision=3, suppress=True, linewidth=220)
src = tools.load('../source_weights_L2.npz')
M = mapping(ns_scale=0.93)
S = tools.states(); X = S['obs'].astype(float)

def unstick(lanes2, kappa):
    M2 = M.copy(); c = np.zeros(46)
    for l in lanes2:
        for k in 'qd':
            M2[I2[f'{k}_{l}'], IW['t_green']] += kappa
            c[I2[f'{k}_{l}']] -= 0.2 * kappa
    w = build(M2, src); w['b1'] = src['b1'] + src['W1'] @ c
    return w

def stuck_count(net, maxcars=2):
    lanes = list(SERVE); bad = []; picks_wrong = 0; tot = 0
    configs = [{l: n} for n in range(1, maxcars + 1) for l in lanes] + \
              [{a: 1, b: 1} for a, b in itertools.combinations(lanes, 2)]
    for cars in configs:
        need = {SERVE[l] for l in cars}
        for p in range(4):
            if p in need: continue
            tot += 1
            a = int(tools.greedy(net, x_w(cars, p, 60))[0])
            if a == p: bad.append((tuple(cars.items()), p))
            elif a not in need: picks_wrong += 1
    return len(bad), picks_wrong, tot, bad

NONP2 = ['N0', 'N1', 'N2', 'S0', 'S1', 'S2', 'E2', 'W2']  # L2 names: E2/W2 = left pockets
ALL = NONP2 + ['E0', 'E1', 'W0', 'W1']
base = plain()
ok = X[:, 52] > .5; tg = np.round(X[:, 53] * 60)
a0 = tools.greedy(base, X)
nb, nw, tot, _ = stuck_count(base)
print(f'delivered(no unstick): stuck {nb}/{tot}, switched to a phase serving none {nw}')
for nm, lanes in [('U1 non-phase2 lanes', NONP2), ('U2 all lanes', ALL)]:
    for kappa in [0.5, 1.0, 1.5]:
        w = unstick(lanes, kappa)
        nb, nw, tot, bad = stuck_count(w)
        a = tools.greedy(w, X)
        same12 = (a[ok & (tg == 12)] == a0[ok & (tg == 12)]).mean()
        chg_long = (a[ok & (tg > 12)] != a0[ok & (tg > 12)]).mean()
        tr = tools.trainability(w)
        print(f'{nm:20s} k={kappa}: stuck {nb}/{tot}, wrong-phase switch {nw}; tg=12 same {same12:.3f}; '
              f'recorded tg>12 changed {chg_long:.3f}; medmaxp {tr["median_max_prob"]:.4f} sat_l1 {tr["sat_l1"]:.4f}')
        if kappa == 1.0: print('    remaining stuck:', bad[:6])
