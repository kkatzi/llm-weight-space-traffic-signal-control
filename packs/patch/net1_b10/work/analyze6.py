import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); A = S['act']; ep = S['episode']; t = S['time_s']; c = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
capS = 22.4/7.5; capU = 256/7.5
q = X[:, 0:12]*capS; d = X[:, 12:24]*capS; qu = X[:, 24:32]*capU; du = X[:, 32:40]*capU
g = tools.greedy(w, X)
np.set_printoptions(precision=3, suppress=True, linewidth=150)
appr = ['N','E','S','W']
for ci in range(3):
    m = c == ci
    print(S['controller_names'][ci])
    for k, a in enumerate(appr):
        full = X[m, 3*k+2] > 0.99
        up1 = qu[m, 2*k+1]
        up0 = qu[m, 2*k]
        print(f'  {a}: pocket full {full.mean():.3f}; q upstream lane1>0 {(up1>0).mean():.3f} mean {up1.mean():.2f}; lane0 >0 {(up0>0).mean():.3f} mean {up0.mean():.2f}; pocket full & up1 q>0: {(full & (up1>0)).mean():.3f}')
    print('  total upstream queue veh mean', qu[m].sum(1).mean(), ' stopline q veh mean', q[m].sum(1).mean(), ' stopline d mean', d[m].sum(1).mean(), ' up d mean', du[m].sum(1).mean())
