# Exploration script (run during design). "plain()" = lane mapping without the unstick term.
import numpy as np
from common import *
from unstick_check import report
src = tools.load('../source_weights_L2.npz')
M = mapping(ns_scale=0.93)
S = tools.states(); X = S['obs'].astype(float); ok = X[:, 52] > .5; tg = np.round(X[:, 53] * 60)

def unstick(lanes2, kappa):
    M2 = M.copy(); c = np.zeros(46)
    for l in lanes2:
        for k in 'qd':
            M2[I2[f'{k}_{l}'], IW['t_green']] += kappa
            c[I2[f'{k}_{l}']] -= 0.2 * kappa
    w = build(M2, src); w['b1'] = src['b1'] + src['W1'] @ c
    return w
NONP2 = ['N0', 'N1', 'N2', 'S0', 'S1', 'S2', 'E2', 'W2']
ALL = NONP2 + ['E0', 'E1', 'W0', 'W1']
NS = ['N0', 'N1', 'N2', 'S0', 'S1', 'S2']
base = plain(); a0 = tools.greedy(base, X)
report(base, 'delivered (no unstick)')
for nm, lanes in [('U1 non-ph2', NONP2), ('U2 all', ALL), ('U3 NS only', NS)]:
    for k in [0.25, 0.5, 1.0, 1.5, 2.0]:
        w = unstick(lanes, k)
        a = tools.greedy(w, X)
        report(w, f'{nm} k={k} (rec tg>12 chg {(a[ok&(tg>12)]!=a0[ok&(tg>12)]).mean():.3f})')
