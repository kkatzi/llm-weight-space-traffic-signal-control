"""Probe: near-empty junction, only W2 pocket (invisible) and W1u upstream occupied. What does a network pick?"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools
from build import build

def state(phase, tg, nW1u, ok=1.0, w1u_moving=False):
    x = np.zeros(46)
    x[40 + phase] = 1; x[44] = ok; x[45] = min(1, tg / 60)
    if nW1u:
        if not w1u_moving:
            x[31] = nW1u / 34.13
        x[39] = nW1u / 34.13
    return x

def run(w, label):
    print(label)
    for n in [0, 1, 2, 3, 5, 8]:
        row = []
        for p in range(4):
            for tg in [12, 30, 60]:
                row.append(int(tools.greedy(w, state(p, tg, n))[0]))
        print(f'  W1u halting={n}:', ' '.join(f'p{p}:' + ''.join(str(r) for r in row[p*3:(p+1)*3]) for p in range(4)))

if __name__ == '__main__':
    w0 = tools.load('start_weights.npz')
    run(w0, 'start (picks for t_green 12,30,60 s per current phase)')
    for spec in sys.argv[1:]:
        run(build(json.loads(spec), w0), spec)
