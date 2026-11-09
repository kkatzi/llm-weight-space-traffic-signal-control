"""Frozen-snapshot switching check (no traffic model): vehicles are frozen, only the signal timing rule is iterated.
From each phase that does not serve the waiting car(s), starting at t_green = 12 s: if the net keeps the phase,
t_green += 5 s; if it picks another phase, that phase starts (3 s yellow, next honoured decision 15 s later).
Report how long until a phase serving the cars is picked (cap 30 min)."""
import itertools, sys
import numpy as np
from common import *

SERVE = {'N0': 0, 'N1': 0, 'S0': 0, 'S1': 0, 'N2': 1, 'S2': 1, 'E0': 2, 'E1': 2, 'E2': 2, 'W0': 2, 'W1': 2, 'W2': 2,
         'E3': 3, 'W3': 3}

def x_w(cars, p, tg, up=0.0):
    x = np.zeros(54)
    for l, n in cars.items():
        cap = 19.2 / 7.5 if l[0] in 'NS' else 22.4 / 7.5
        x[IW['q_' + l]] = x[IW['d_' + l]] = min(n / cap, 1)
    x[48 + p] = 1; x[52] = 1; x[53] = min(tg, 60) / 60
    return x

def time_to_serve(net, cars, p0, cap_s=1800):
    need = {SERVE[l] for l in cars}
    p, tg, t = p0, 12, 0
    while t < cap_s:
        a = int(tools.greedy(net, x_w(cars, p, tg))[0])
        if a in need:
            return t
        if a == p:
            tg += 5; t += 5
        else:
            p, tg, t = a, 12, t + 15
    return cap_s

def configs():
    lanes = list(SERVE)
    return [{l: n} for n in (1, 2) for l in lanes] + [{a: 1, b: 1} for a, b in itertools.combinations(lanes, 2)]

def report(net, label):
    res = []
    for cars in configs():
        need = {SERVE[l] for l in cars}
        for p in range(4):
            if p not in need:
                res.append(time_to_serve(net, cars, p))
    res = np.array(res)
    print(f'{label:34s} cases {len(res)}  never served (30 min) {int((res >= 1800).sum()):3d}  '
          f'mean wait (served) {res[res < 1800].mean():5.1f} s  max (served) {res[res < 1800].max():4d} s')
    return res

if __name__ == '__main__':
    report(tools.load(os.path.join(HERE, 'weights.npz')), 'delivered')
