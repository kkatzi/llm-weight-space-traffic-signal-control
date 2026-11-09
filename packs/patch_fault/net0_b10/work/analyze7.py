import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
np.set_printoptions(precision=3, suppress=True, linewidth=200)
S = tools.states(); X = S['obs'].astype(np.float64); C = S['controller']; T = S['time_s']
ok = X[:, 44] > 0.5; cur = X[:, 40:44].argmax(1)
wp = tools.load('work/proxy_a1.npz'); gp = tools.greedy(wp, X)
h1, _ = tools.hidden(wp, X)
m_net = ok & (C == 0)
main = (T < 3600)
print('ph2 tg>=0.99 states: net main-hour', int((m_net & main & (cur==2) & (X[:,45]>0.99)).sum()), 'net drain', int((m_net & ~main & (cur==2) & (X[:,45]>0.99)).sum()))
print('all ctrls tg>=0.99 any phase:', [int((ok & (C==c) & (X[:,45]>0.99) & main).sum()) for c in range(3)])
# importance of each L1 unit, replacing it by a constant (mean) -- on net states and on all states, main hour
res = []
for j in range(64):
    w = {k: v.copy() for k, v in wp.items()}
    mu = h1[C == 0, j].mean()
    w['b2'] = w['b2'] + w['W2'][:, j] * mu; w['W2'][:, j] = 0
    g = tools.greedy(w, X)
    lg0 = tools.forward(wp, X); lg = tools.forward(w, X)
    res.append((int((g[m_net & main] != gp[m_net & main]).sum()), int((g[ok & main] != gp[ok & main]).sum()), j, round(mu, 3),
                round(float(np.abs(lg - lg0)[m_net].mean()), 3)))
res.sort()
print('(changed net main-hour, changed all main-hour, unit, mean, mean |dlogit|)')
for r in res[:10]: print(r)
