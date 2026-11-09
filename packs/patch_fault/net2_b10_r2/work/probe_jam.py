"""Probe jam states: which phase does a network pick when lanes are saturated?"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools
from build import build

def jam(phase, tg, level=1.0, w_only=None):
    x = np.full(46, level); x[40:46] = 0
    x[40 + phase] = 1; x[44] = 1; x[45] = min(1, tg / 60)
    x[11] = 0; x[23] = 0
    return x

def run(w, label, nets_inputs):
    out = []
    for name, f in nets_inputs:
        row = ' '.join(f'p{p}:' + ''.join(str(int(tools.greedy(w, f(p, tg))[0])) for tg in [12, 30, 60]) for p in range(4))
        out.append(f'  {name:28s} {row}')
    print(label); print('\n'.join(out))

def make_inputs():
    S = tools.states(); X = S['obs'].astype(float)
    hi = np.percentile(X, 90, axis=0)
    def full(p, tg): return jam(p, tg, 1.0)
    def half(p, tg): return jam(p, tg, 0.5)
    def stops_full_up_low(p, tg):
        x = jam(p, tg, 1.0); x[24:40] = 0.1; return x
    def w_jam_phase3(p, tg):   # W straight blocked: W1 full, W1u full, E side mild, NS mild
        x = jam(p, tg, 0.3); x[24:40] = 0.05
        x[9] = x[10] = x[21] = x[22] = 1; x[31] = x[39] = 1; x[30] = x[38] = 0.5
        x[3] = x[4] = x[15] = x[16] = 1; x[27] = x[35] = 0.5
        x[5] = x[17] = 0
        return x
    return [('all saturated', full), ('all at 0.5', half), ('stop lines full, upstream 0.1', stops_full_up_low),
            ('EW through jammed, E2 empty', w_jam_phase3)]

if __name__ == '__main__':
    w0 = tools.load('start_weights.npz')
    ins = make_inputs()
    run(w0, 'start', ins)
    for spec in sys.argv[1:]:
        run(build(json.loads(spec), w0), spec, ins)
