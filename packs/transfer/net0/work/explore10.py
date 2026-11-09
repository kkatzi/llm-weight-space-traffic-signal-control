# Exploration script (run during design). "plain()" = lane mapping without the unstick term.
import numpy as np
from common import *
from unstick_check import configs, time_to_serve, SERVE, report
from proxy import score
import explore8 as e8
S = tools.states(); X = S['obs'].astype(float); ok = X[:, 52] > .5; tg = np.round(X[:, 53] * 60)
base = plain(); a0 = tools.greedy(base, X); m = ok & (tg > 12)
for k in [0.5, 0.75, 1.0]:
    w = e8.unstick(e8.NONP2, k)
    a = tools.greedy(w, X)
    report(w, f'k={k}')
    print('    proxy tg>12', round(score(X[m], a[m])['served/best'], 3), ' changed', round(float((a[m] != a0[m]).mean()), 3))
    bad = [(cars, p) for cars in configs() for p in range(4) if p not in {SERVE[l] for l in cars}
           and time_to_serve(w, cars, p) >= 1800]
    print('    never served:', bad)
