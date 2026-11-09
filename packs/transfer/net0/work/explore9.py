# Exploration script (run during design). "plain()" = lane mapping without the unstick term.
import numpy as np
from common import *
from proxy import score
from explore8 import unstick, NONP2
np.set_printoptions(precision=3, suppress=True, linewidth=220)
src = tools.load('../source_weights_L2.npz')
S = tools.states(); X = S['obs'].astype(float); ok = X[:, 52] > .5; tg = np.round(X[:, 53] * 60)
base = plain()
m = ok & (tg > 12)
for k in [0.5, 1.0]:
    w = unstick(NONP2, k)
    print(f'k={k} recorded L2W ok states with tg>12 (n={m.sum()}): proxy base', score(X[m], tools.greedy(base, X[m])),
          '\n        unstick', score(X[m], tools.greedy(w, X[m])))
# the real stuck L2 state, unstick applied in L2 space
L2 = tools.states('states_L2.npz'); X2 = L2['obs'].astype(float)
st = X2[(L2['episode'] == 2) & (L2['time_s'] == 4500)][0]
for k in [0.5, 1.0]:
    seq = []
    for t in [12, 17, 22, 27, 32, 37, 42, 60]:
        x = st.copy(); x[45] = t / 60
        for l in NONP2:
            for q in 'qd':
                x[I2[f'{q}_{l}']] += k * (t / 60 - 0.2)
        seq.append((t, int(tools.greedy(src, x)[0])))
    print(f'k={k} real L2 stuck state (green phase 2; cars N0,N1,W-left): pick by t_green', seq)
# also the L2W analog of that state from every phase
from unstick_check import x_w
for k in [0.0, 1.0]:
    w = base if k == 0 else unstick(NONP2, k)
    print(f'k={k} L2W analog N0+N1+W3 at phase 2:', [(t, int(tools.greedy(w, x_w({"N0": 1, "N1": 1, "W3": 1}, 2, t))[0])) for t in [12, 17, 22, 27, 32, 60]])
