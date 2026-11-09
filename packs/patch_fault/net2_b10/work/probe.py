"""Probe a network on static 'lone vehicle' states: emulate the phase logic (5 s decisions, 10 s min green,
3 s yellow) with a frozen queue and report whether the lane's serving phase is ever given."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools

names = tools.names()
SERVE = {'N0': 0, 'N1': 0, 'S0': 0, 'S1': 0, 'N2': 1, 'S2': 1, 'E0': 2, 'E1': 2, 'W0': 2, 'W1': 2, 'E2': 3, 'W2': 3}
LANES = ['N0', 'N1', 'N2', 'E0', 'E1', 'E2', 'S0', 'S1', 'S2', 'W0', 'W1', 'W2']


def obs(phase, tg, base):
    x = base.copy()
    x[40:44] = 0; x[40 + phase] = 1
    x[44] = 1.0 if tg >= 10 else 0.0
    x[45] = min(tg / 60.0, 1.0)
    return x


def run(w, base, start_phase, target, steps=200, fault=True):
    ph, tg = start_phase, 20.0
    served_at = None
    for k in range(steps):
        x = obs(ph, tg, base)
        if fault:
            x[11] = 0; x[23] = 0
        a = int(tools.greedy(w, x)[0])
        if a != ph and tg >= 10:
            ph, tg = a, 2.0
        else:
            tg += 5
        if ph == target and served_at is None:
            served_at = k
            return served_at
    return None


def lone_states(nveh=1):
    out = []
    for L in LANES:
        i = names.index('q_' + L); j = names.index('d_' + L)
        b = np.zeros(46); b[i] = nveh / 3.0 * 1.005; b[j] = nveh / 3.0 * 1.005
        b = np.clip(b, 0, 1)
        out.append((L, b))
    return out


def report(w, fault=True, nveh=1):
    bad = []
    for L, b in lone_states(nveh):
        if fault and L == 'W2':
            continue
        for p in range(4):
            r = run(w, b, p, SERVE[L], fault=fault)
            if r is None:
                bad.append((L, p))
    return bad


if __name__ == '__main__':
    w = tools.load(sys.argv[1] if len(sys.argv) > 1 else 'start_weights.npz')
    for n in (1, 2, 3):
        print('nveh', n, 'deadlocks (lane, start phase):', report(w, True, n))
