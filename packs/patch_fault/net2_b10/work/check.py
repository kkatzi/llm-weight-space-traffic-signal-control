"""check.py: offline checks of a candidate (forward passes only)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK); sys.path.insert(0, HERE)
import numpy as np
import tools
from probe import obs, SERVE, LANES

names = tools.names(); ix = {n: i for i, n in enumerate(names)}


def trace(w, base, p0, steps=60, tg0=20.0):
    ph, tg = p0, tg0; seq = []
    for _ in range(steps):
        x = obs(ph, tg, base); x[11] = 0; x[23] = 0
        a = int(tools.greedy(w, x)[0])
        if a != ph and tg >= 10:
            ph, tg = a, 2.0
        else:
            tg += 5
        seq.append(ph)
    return seq


def deadlocks(w):
    bad = []
    for n in (1, 2, 3):
        for L in LANES:
            if L == 'W2':
                continue
            b = np.zeros(46); b[ix['q_' + L]] = min(n * 0.335, 1); b[ix['d_' + L]] = min(n * 0.335, 1)
            for p in range(4):
                if p == SERVE[L]:
                    continue
                if SERVE[L] not in trace(w, b, p):
                    bad.append((L, n, p))
    # empty junction (a hidden W2 car would need phase 3)
    for p in range(4):
        if 3 not in trace(w, np.zeros(46), p):
            bad.append(('empty->3', 0, p))
    for n in (1, 2, 3, 4, 6, 8):
        b = np.zeros(46); b[ix['q_W1u']] = n / 34.13; b[ix['d_W1u']] = n / 34.13
        for p in range(4):
            if 3 not in trace(w, b, p):
                bad.append(('W1u', n, p))
    return bad


def report(w, w0, label=''):
    S = tools.states(); X = S['obs'].astype(np.float64); t = S['time_s']
    ok = X[:, 44] > 0.5
    a0 = tools.greedy(w0, X); a = tools.greedy(w, X)
    Xi = X.copy(); Xi[:, 11] = X[:, 5]; Xi[:, 23] = X[:, 17]
    am = tools.greedy(w0, Xi)          # original network fed the mirror imputation explicitly
    print(label, tools.edit_size(w, w0))
    print('  agree with start (ok states): %.3f   with start+mirror-input: %.3f' % ((a[ok] == a0[ok]).mean(), (a[ok] == am[ok]).mean()))
    main = ok & (t < 3600)
    print('  agree with start+mirror-input, main hour: %.3f' % (a[main] == am[main]).mean())
    print('  phase choice share (ok):', np.bincount(a[ok], minlength=4) / ok.sum())
    print('  deadlocks:', deadlocks(w))
    print('  empty traces:', [''.join(map(str, trace(w, np.zeros(46), p, 20))) for p in range(4)])
    print('  trainability:', {k: round(v, 3) for k, v in tools.trainability(w).items()})


if __name__ == '__main__':
    w0 = tools.load(os.path.join(PACK, 'start_weights.npz'))
    for p in sys.argv[1:]:
        report(tools.load(p), w0, p)
