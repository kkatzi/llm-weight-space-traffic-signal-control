"""table.py: static decision tables on synthetic lone-vehicle states (forward passes only)."""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import tools
NM = tools.names()

def state(lane, p, tg, n=1):
    x = np.zeros(46)
    if lane is not None and lane != 11:
        x[lane] = 0.335 * n; x[12 + lane] = 0.335 * n
    x[40 + p] = 1; x[44] = float(tg >= 10); x[45] = min(tg / 60, 1)
    return x

def table(w, label, tgs=(15, 30, 60)):
    print(f'--- {label}: chosen phase for one halted car in lane (rows) when green phase p has lasted t s')
    print('lane    ' + ' '.join(f'p{p}@{t:<3d}' for p in range(4) for t in tgs))
    for lane in [None] + list(range(12)):
        row = []
        for p in range(4):
            for t in tgs:
                row.append(str(int(tools.greedy(w, state(lane, p, t))[0])).center(6))
        print(f'{(NM[lane] if lane is not None else "empty"):7s} ' + ' '.join(row))

if __name__ == '__main__':
    from build import cand, W0
    table(W0, 'start')
    for nm in sys.argv[1:]:
        table(cand(nm), nm)
