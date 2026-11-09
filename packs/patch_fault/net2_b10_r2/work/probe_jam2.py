"""Parametric jam probe: stop lines full (except the lanes the current phase just drained), upstream lanes at level L,
W1u at L+delta. Prints whether the network leaves the current phase (at t_green 12/30/60 s)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools
from build import build

# stop-line lanes served by each phase (indices into 0..11)
SERVED = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}

def jam_state(p, tg, L, delta, stop=1.0):
    x = np.zeros(46)
    x[0:12] = stop; x[12:24] = stop
    for i in SERVED[p]:
        x[i] = 0.0; x[12 + i] = 0.3       # served lanes: moving, not halting
    x[24:32] = L; x[32:40] = min(1, L + 0.05)
    x[31] = min(1, L + delta); x[39] = min(1, L + delta + 0.05)
    x[11] = 0; x[23] = 0
    x[40 + p] = 1; x[44] = 1; x[45] = min(1, tg / 60)
    return x

def table(w):
    rows = []
    for p in range(4):
        cells = []
        for L in [0.0, 0.1, 0.3, 0.6, 1.0]:
            for dl in [0.0, 0.1, 0.3]:
                if L + dl > 1.0001: cells.append('---'); continue
                cells.append(''.join(str(int(tools.greedy(w, jam_state(p, tg, L, dl))[0])) for tg in [12, 30, 60]))
        rows.append(f'  p{p}: ' + ' '.join(cells))
    return '\n'.join(rows)

if __name__ == '__main__':
    w0 = tools.load('start_weights.npz')
    print('columns: L=0,0.1,0.3,0.6,1.0 x delta=0,0.1,0.3 ; entries = picks at tg 12/30/60')
    print('start'); print(table(w0))
    for spec in sys.argv[1:]:
        print(spec); print(table(build(json.loads(spec), w0)))
