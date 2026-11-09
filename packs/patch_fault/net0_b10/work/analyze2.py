import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w0 = tools.load('start_weights.npz'); wp = tools.load('work/proxy_a1.npz')
S = tools.states(); X = S['obs'].astype(np.float64); C = S['controller']; T = S['time_s']; E = S['episode']
print(tools.trainability(wp))
# empty states
for ph in range(4):
    for tg in [2/60, 7/60, 12/60, 0.5, 1.0]:
        x = np.zeros(46); x[40 + ph] = 1; x[44] = 1.0 if tg >= 10/60 else 0.0; x[45] = tg
        lg = tools.forward(wp, x)[0]
        print('empty ph', ph, 'tg', round(tg, 2), 'logits', lg, 'argmax', lg.argmax())
# drain states in recorded data: total obs load small
load = X[:, :40].sum(1)
print('load quantiles', np.quantile(load, [0, .01, .05, .1, .5]))
m = load < 1e-6
print('n empty states', m.sum(), 'controllers', np.bincount(C[m], minlength=3))
g = tools.greedy(wp, X)
print('proxy choices in empty+ok states', np.bincount(g[m & (X[:,44]>0.5)], minlength=4))
# how the start network behaves in the network's own episodes at the end
for e in np.unique(E[C==0])[:6]:
    mm = (C==0)&(E==e)
    print('ep', e, 'n', mm.sum(), 'max t', T[mm].max())
