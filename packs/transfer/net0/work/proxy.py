"""Heuristic sanity proxy (no fitting): how much waiting traffic does the chosen phase serve?"""
import numpy as np
from common import *

CAP_W = {}
for a in 'NESW':
    lanes = '012' if a in 'NS' else '0123'
    for l in lanes:
        CAP_W[f'{a}{l}'] = (19.2 if a in 'NS' else 22.4) / 7.5
    for l in ('01' if a in 'NS' else '012'):
        CAP_W[f'{a}{l}u'] = 256 / 7.5
SERVE_W = {0: {'N0': 1, 'N1': 1, 'S0': 1, 'S1': 1, 'N0u': 1, 'S0u': 1, 'N1u': .5, 'S1u': .5},
           1: {'N2': 1, 'S2': 1, 'N1u': .5, 'S1u': .5},
           2: {'E0': 1, 'E1': 1, 'E2': 1, 'W0': 1, 'W1': 1, 'W2': 1, 'E0u': 1, 'E1u': 1, 'W0u': 1, 'W1u': 1,
               'E2u': .5, 'W2u': .5},
           3: {'E3': 1, 'W3': 1, 'E2u': .5, 'W2u': .5}}
CAP_2 = {}
for a in 'NESW':
    for l in '012':
        CAP_2[f'{a}{l}'] = 22.4 / 7.5
    for l in '01':
        CAP_2[f'{a}{l}u'] = 256 / 7.5
SERVE_2 = {0: {'N0': 1, 'N1': 1, 'S0': 1, 'S1': 1, 'N0u': 1, 'S0u': 1, 'N1u': .5, 'S1u': .5},
           1: {'N2': 1, 'S2': 1, 'N1u': .5, 'S1u': .5},
           2: {'E0': 1, 'E1': 1, 'W0': 1, 'W1': 1, 'E0u': 1, 'W0u': 1, 'E1u': .5, 'W1u': .5},
           3: {'E2': 1, 'W2': 1, 'E1u': .5, 'W1u': .5}}


def phase_load(X, kind='q', l2=False):
    idx, cap, serve = (I2, CAP_2, SERVE_2) if l2 else (IW, CAP_W, SERVE_W)
    out = np.zeros((len(X), 4))
    for p, d in serve.items():
        for lane, f in d.items():
            out[:, p] += f * X[:, idx[f'{kind}_{lane}']] * cap[lane]
    return out


def score(X, a, l2=False):
    ok = X[:, (44 if l2 else 52)] > .5
    q = phase_load(X, 'q', l2) + 0.5 * phase_load(X, 'd', l2)
    chosen = q[np.arange(len(X)), a]
    best = q.max(1)
    m = ok & (best > 0)
    return {'served/best': float(chosen[m].sum() / best[m].sum()),
            'pick_best': float((chosen[m] >= best[m] - 1e-9).mean()),
            'wasted': float((chosen[m] == 0).mean()),
            'n': int(m.sum())}
