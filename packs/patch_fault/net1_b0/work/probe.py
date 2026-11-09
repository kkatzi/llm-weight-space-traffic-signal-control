"""Probe a weight set on synthetic sparse states: which phase does it pick?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
N = tools.names(); IX = {n: i for i, n in enumerate(N)}
SERVE = {0: ['N0','N1','S0','S1'], 1: ['N2','S2'], 2: ['E0','E1','W0','W1'], 3: ['E2','W2']}
LANE_PHASE = {l: p for p, ls in SERVE.items() for l in ls}

def state(phase, tg, ok=1, lanes=(), up=(), qv=1/3, dv=None, upd=1/34.1):
    x = np.zeros(46)
    x[40 + phase] = 1; x[44] = ok; x[45] = tg
    for l in lanes:
        x[IX['q_' + l]] = qv; x[IX['d_' + l]] = qv if dv is None else dv
    for u in up:
        x[IX['d_' + u]] = upd
    return x

def probe(w, fill=None, verbose=True):
    """fill(X) -> X with x11,x23 filled (what the deployed net sees is always 0 there; fill is only for
    probing an input-substitution idea without editing weights)."""
    f = (lambda X: X) if fill is None else fill
    out = {}
    rows = []
    for p in range(4):
        for tg in [0.2, 0.5, 1.0]:
            X = np.array([state(p, tg)])
            rows.append(('empty', p, tg, int(tools.greedy(w, f(X))[0])))
    for l in [k for k in LANE_PHASE if k != 'W2']:
        for p in range(4):
            for tg in [0.2, 1.0]:
                X = np.array([state(p, tg, lanes=[l])])
                a = int(tools.greedy(w, f(X))[0])
                rows.append((l, p, tg, a))
    if verbose:
        for r in rows:
            mark = '' if r[0] == 'empty' or r[3] == LANE_PHASE[r[0]] else ('  <-- holds' if r[3] == r[1] else '  <-- other')
            print(r, mark)
    return rows

if __name__ == '__main__':
    w = tools.load(sys.argv[1] if len(sys.argv) > 1 else 'start_weights.npz')
    probe(w)
