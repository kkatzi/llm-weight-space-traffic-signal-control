"""Policy probe: with one visible halted car on a single lane (everything else empty), step through the
network's decisions (every 5 s; switch honoured only when ok; new green is 2 s old at the next decision) and
report how many seconds pass until a phase that serves the lane is green, or 'STUCK' if the decision sequence
reaches a fixed point that never serves it. Pure forward passes on synthetic inputs; no traffic model."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools

n = tools.names(); I = {nm: i for i, nm in enumerate(n)}
SERVES = {0: ['N0', 'N1', 'S0', 'S1'], 1: ['N2', 'S2'], 2: ['E0', 'E1', 'W0', 'W1'], 3: ['E2', 'W2']}
LANES = ['N0', 'N1', 'N2', 'E0', 'E1', 'E2', 'S0', 'S1', 'S2', 'W0', 'W1', 'W2']


def served(lane, p):
    return lane in SERVES[p]


def wait_until_served(w, lane, p0, cars=1):
    p, age = p0, 12.0
    for step in range(400):
        if served(lane, p) and age >= 2:
            return step * 5
        x = np.zeros(46); x[40 + p] = 1; x[45] = min(age / 60, 1); x[44] = 1.0 if age >= 10 else 0.0
        if lane != 'W2':              # W2 car is invisible (failed detector)
            x[I['q_' + lane]] = x[I['d_' + lane]] = min(cars / 2.99, 1)
        a = int(tools.forward(w, x)[0].argmax())
        if a != p and age >= 10:
            p, age = a, 2.0           # 3 s yellow, next decision sees a 2 s old green
        else:
            age += 5
    return None


def report(w, label):
    print(label)
    for cars in (1, 3):
        rows = []
        for lane in LANES:
            r = [wait_until_served(w, lane, p0, cars) for p0 in range(4)]
            rows.append('%s:%s' % (lane, '/'.join('STUCK' if v is None else str(v) for v in r)))
        print('  cars=%d ' % cars + '  '.join(rows))


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(HERE, 'work'))
    import build_v1, build_v2
    report(build_v1.build()[1], 'v1')
    report(build_v2.build()[1], 'v2')
