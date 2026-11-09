import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w0 = tools.load('start_weights.npz')
S = tools.states(); X = S['obs'].astype(np.float64); C = S['controller']; T = S['time_s']; E = S['episode']
names = tools.names()
ok = X[:,44] > 0.5; cur = X[:,40:44].argmax(1); load = X[:, :40].sum(1)
m = ok & (cur == 2) & (X[:,45] > 0.99) & (C == 0)
print('episodes of long holds', np.unique(E[m], return_counts=True))
nz = X[m][:, :40]
print('mean of nonzero feature in stuck states:')
for i in np.argsort(-nz.mean(0))[:8]: print(names[i], nz[:, i].mean().round(3), (nz[:, i] > 0).mean().round(3))
for e in np.unique(E[m]):
    mm = (C==0)&(E==e)
    t0 = T[m & (E==e)].min()
    print('ep', e, 'first long-hold t', t0, 'last t', T[mm].max())
    # print a few states around
    idx = np.where(mm & (T >= t0 - 30) & (T <= t0 + 10))[0]
    for i in idx:
        b = {n: v for n, v in tools.block(X[i]).items() if v != 0}
        print('  t', T[i], 'act', S['act'][i], b)
# single E-left car in empty network, ph2
for tg in [0.2, 0.5, 1.0]:
    x = np.zeros(46); x[42] = 1; x[44] = 1; x[45] = tg; x[5] = 1/2.9867; x[17] = 1/2.9867
    print('empty+1 E-left car, ph2, tg', tg, tools.forward(w0, x)[0].round(2))
    x = np.zeros(46); x[42] = 1; x[44] = 1; x[45] = tg; x[11] = 1/2.9867; x[23] = 1/2.9867
    print('empty+1 W-left car (full info), ph2, tg', tg, tools.forward(w0, x)[0].round(2))
