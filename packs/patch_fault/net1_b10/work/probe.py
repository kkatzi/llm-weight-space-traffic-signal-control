"""Probe helper: decisions on synthetic drain-like states and change rates on recorded states."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "work"))
import tools
from build_v1 import build

n = tools.names(); I = {nm: i for i, nm in enumerate(n)}
CASES = {'empty': {}, 'W1u 2 cars': {I['q_W1u']: 0.059, I['d_W1u']: 0.059},
         'N2 1car': {I['q_N2']: 0.335, I['d_N2']: 0.335}, 'S2 1car': {I['q_S2']: 0.335, I['d_S2']: 0.335},
         'N0 1car': {I['q_N0']: 0.335, I['d_N0']: 0.335}, 'E0 1car': {I['q_E0']: 0.335, I['d_E0']: 0.335},
         'W1 1car': {I['q_W1']: 0.335, I['d_W1']: 0.335}, 'E2 1car': {I['q_E2']: 0.335, I['d_E2']: 0.335},
         'N1 2car': {I['q_N1']: 0.67, I['d_N1']: 0.67}}
TGS = [0.2, 0.4, 0.6, 0.8, 1.0]


def st(p, tg, extra):
    x = np.zeros(46); x[40 + p] = 1; x[45] = tg; x[44] = 1.0
    for i, v in extra.items(): x[i] = v
    return x


def picks(w, ex):
    return [''.join(str(tools.forward(w, st(p, tg, ex))[0].argmax()) for tg in TGS) for p in range(4)]


def report(w, wref, label=''):
    S = tools.states(); X = S['obs'].astype(float); c = S['controller']
    ph = X[:, 40:44].argmax(1)
    real = (c > 0) & (X[:, 44] > 0.5) & (X[:, 45] <= 0.55)
    a1 = tools.greedy(wref, X); ak = tools.greedy(w, X)
    chg = {p: round(float((ak[real & (ph == p)] != a1[real & (ph == p)]).mean()), 3) for p in range(4)}
    print(label, 'changed vs ref (tg<=.55) by phase', chg)
    for nm, ex in CASES.items():
        print('   %-11s ref %s  new %s' % (nm, picks(wref, ex), picks(w, ex)))
