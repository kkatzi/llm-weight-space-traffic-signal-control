# Exploration script (run during design). "plain()" = lane mapping without the unstick term.
"""Lone-vehicle probes: which waiting vehicles does the net serve when nothing else is around?"""
import numpy as np, itertools
from common import *
src = tools.load('../source_weights_L2.npz')
w = plain()
SERVE = {'N0': 0, 'N1': 0, 'S0': 0, 'S1': 0, 'N2': 1, 'S2': 1, 'E0': 2, 'E1': 2, 'E2': 2, 'W0': 2, 'W1': 2, 'W2': 2, 'E3': 3, 'W3': 3}
SERVE2 = {'N0': 0, 'N1': 0, 'S0': 0, 'S1': 0, 'N2': 1, 'S2': 1, 'E0': 2, 'E1': 2, 'W0': 2, 'W1': 2, 'E2': 3, 'W2': 3}

def x_w(cars, p, tg=12):
    x = np.zeros(54)
    for l, n in cars.items():
        cap = 19.2 / 7.5 if l[0] in 'NS' else 22.4 / 7.5
        x[IW['q_' + l]] = x[IW['d_' + l]] = min(n / cap, 1)
    x[48 + p] = 1; x[52] = 1; x[53] = tg / 60
    return x

def x_2(cars, p, tg=12):
    x = np.zeros(46)
    for l, n in cars.items():
        x[I2['q_' + l]] = x[I2['d_' + l]] = min(n / (22.4 / 7.5), 1)
    x[40 + p] = 1; x[44] = 1; x[45] = tg / 60
    return x

def stuck_from(net, mk, serve, cars):
    """Phases from which the net, at tg=60, keeps a green that serves none of the waiting cars (stuck)."""
    need = {serve[l] for l in cars}
    out = []
    for p in range(4):
        a = int(tools.greedy(net, mk(cars, p, 60))[0])
        if a == p and p not in need:
            out.append(p)
    return out

for label, net, mk, serve in [('source on L2', src, x_2, SERVE2), ('delivered on L2W', w, x_w, SERVE)]:
    print(label)
    lanes = list(serve)
    tot = 0; bad = 0
    for n in (1, 2):
        for l in lanes:
            s = stuck_from(net, mk, serve, {l: n}); tot += 1; bad += bool(s)
            if s: print(f'   {n} car(s) on {l}: stuck when green is phase {s}')
    for l1, l2 in itertools.combinations(lanes, 2):
        s = stuck_from(net, mk, serve, {l1: 1, l2: 1}); tot += 1; bad += bool(s)
    print(f'   lone 1-2 car & pair configurations with a stuck green: {bad}/{tot}')
