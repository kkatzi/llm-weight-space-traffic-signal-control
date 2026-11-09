import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz')
nm = tools.names(); ix = {n: i for i, n in enumerate(nm)}
capS = 22.4/7.5; capU = 256/7.5
def state(cur, halting=None, up=None, dens=None, updens=None, tg=12):
    x = np.zeros(46)
    x[40+cur] = 1; x[44] = 1 if tg >= 10 else 0; x[45] = min(tg/60, 1)
    for k, v in (halting or {}).items():
        x[ix['q_'+k]] = min(v/capS, 1); x[ix['d_'+k]] = min(v/capS, 1)
    for k, v in (dens or {}).items():
        x[ix['d_'+k]] = min(v/capS, 1)
    for k, v in (up or {}).items():
        x[ix['q_'+k]] = min(v/capU, 1); x[ix['d_'+k]] = max(x[ix['d_'+k]], min(v/capU, 1))
    for k, v in (updens or {}).items():
        x[ix['d_'+k]] = min(v/capU, 1)
    return x
np.set_printoptions(precision=2, suppress=True, linewidth=160)
def show(desc, x):
    l = tools.forward(w, x)[0]
    print(f'{desc:60s} pick {l.argmax()}  logits {l}')
# empty network
for cur in range(4):
    show(f'empty, cur={cur}', state(cur))
# typical moving traffic upstream
ud = {k: 1.5 for k in ['N0u','N1u','S0u','S1u']}; ud.update({k: 2.0 for k in ['E0u','E1u','W0u','W1u']})
for cur in range(4):
    show(f'moving only, cur={cur}', state(cur, updens=ud))
# single phase waiting
PH = {0: ['N0','N1','S0','S1'], 1: ['N2','S2'], 2: ['E0','E1','W0','W1'], 3: ['E2','W2']}
for cur in range(4):
    for p in range(4):
        if p == cur: continue
        for n in [1, 2, 3]:
            show(f'cur={cur}: phase {p} lanes each {n} halting', state(cur, halting={l: n for l in PH[p]}, updens=ud))
