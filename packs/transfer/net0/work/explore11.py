# Exploration script (run during design). "plain()" = lane mapping without the unstick term.
"""Random sparse frozen snapshots (1-4 waiting cars + some approaching upstream density)."""
import numpy as np
from common import *
from unstick_check import SERVE
import explore8 as e8
rng = np.random.default_rng(0)
UP = ['N0u', 'N1u', 'E0u', 'E1u', 'E2u', 'S0u', 'S1u', 'W0u', 'W1u', 'W2u']
lanes = list(SERVE)
cases = []
for _ in range(600):
    k = rng.integers(1, 5)
    cars = {}
    for l in rng.choice(lanes, size=k):
        cars[l] = cars.get(l, 0) + 1
    up = {u: rng.choice([0, 1, 2]) / 34.13 for u in UP}
    cases.append((cars, up))

def xw(cars, up, p, t):
    x = np.zeros(54)
    for l, n in cars.items():
        cap = 19.2 / 7.5 if l[0] in 'NS' else 22.4 / 7.5
        x[IW['q_' + l]] = x[IW['d_' + l]] = min(n / cap, 1)
    for u, v in up.items():
        x[IW['d_' + u]] = v
    x[48 + p] = 1; x[52] = 1; x[53] = min(t, 60) / 60
    return x

def run(net):
    never = 0; waits = []
    for cars, up in cases:
        need = {SERVE[l] for l in cars}
        for p0 in range(4):
            if p0 in need: continue
            p, t, el = p0, 12, 0
            while el < 1800:
                a = int(tools.greedy(net, xw(cars, up, p, t))[0])
                if a in need: break
                if a == p: t += 5; el += 5
                else: p, t, el = a, 12, el + 15
            if el >= 1800: never += 1
            else: waits.append(el)
    return never, len(waits) + never, np.mean(waits), np.max(waits)

base = plain()
print('delivered (no unstick): never/total, mean wait, max wait', run(base))
for k in [0.5, 0.75, 1.0]:
    print(f'unstick k={k}:', run(e8.unstick(e8.NONP2, k)))

UPSERVE = {'N0u': {0}, 'N1u': {0, 1}, 'S0u': {0}, 'S1u': {0, 1}, 'E0u': {2}, 'E1u': {2}, 'E2u': {2, 3},
           'W0u': {2}, 'W1u': {2}, 'W2u': {2, 3}}
def run2(net):
    never = 0; waits = []; ex = []
    for cars, up in cases:
        need = {SERVE[l] for l in cars}
        need_up = set().union(*[UPSERVE[u] for u, v in up.items() if v > 0]) if any(v > 0 for v in up.values()) else set()
        for p0 in range(4):
            if p0 in need: continue
            p, t, el = p0, 12, 0
            while el < 1800:
                a = int(tools.greedy(net, xw(cars, up, p, t))[0])
                if a in need or (a != p and a in need_up): break   # served, or switched to serve approaching traffic
                if a == p: t += 5; el += 5
                else: p, t, el = a, 12, el + 15
            if el >= 1800:
                never += 1
                if len(ex) < 4: ex.append((cars, p0, p))
            else: waits.append(el)
    return never, len(waits) + never, round(float(np.mean(waits)), 2), int(np.max(waits)), ex
print('--- counting a switch to a phase with approaching traffic as legitimate ---')
print('delivered (no unstick):', run2(base))
for k in [0.5, 0.75, 1.0]:
    print(f'unstick k={k}:', run2(e8.unstick(e8.NONP2, k)))
