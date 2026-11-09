import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load('start_weights.npz')
S = tools.states()
X = S['obs'].astype(float); E = S['episode']; T = S['time_s']; C = S['controller']
names = tools.names()
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
g = tools.greedy(w, X)
d = X[:, 12:24]
D = np.stack([d[:, SERVE[p]].sum(1) for p in range(4)], 1)
q = X[:, :12]
Q = np.stack([q[:, SERVE[p]].sum(1) for p in range(4)], 1)
bad = np.where(ok & (g == ph) & (D[np.arange(len(X)), ph] == 0) & (Q.sum(1) > 0))[0]
print('bad extends (all controllers)', len(bad), 'by phase', np.bincount(ph[bad], minlength=4), 'by ctrl', np.bincount(C[bad], minlength=3))
# all ok states with current phase empty
emp = np.where(ok & (D[np.arange(len(X)), ph] == 0))[0]
print('ok & current empty', len(emp), 'by phase', np.bincount(ph[emp], minlength=4), 'extend share by phase',
      [round(float((g[emp][ph[emp] == p] == p).mean()), 2) if (ph[emp] == p).any() else None for p in range(4)])
# sensitivity for a few bad states: effect of zeroing each feature group on logit[cur]-max(other)
for i in bad[:6]:
    x = X[i].copy(); lg = tools.forward(w, x)[0]; p = ph[i]
    def margin(xx):
        l = tools.forward(w, xx)[0]; return l[p] - np.max(np.delete(l, p))
    base = margin(x)
    eff = []
    for j in range(46):
        if x[j] != 0 and j < 40:
            xx = x.copy(); xx[j] = 0; eff.append((names[j], round(margin(xx) - base, 2)))
    # t_green effect
    for tgv in [12, 17, 22, 27, 40]:
        xx = x.copy(); xx[45] = tgv / 60; eff.append(('tg=%d' % tgv, round(margin(xx) - base, 2)))
    print(C[i], T[i], 'cur', p, 'tg', round(x[45]*60), 'margin', round(base, 2), eff)
