import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); A = S['act']; ep = S['episode']; t = S['time_s']; c = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
capS = 22.4/7.5; capU = 256/7.5
PL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
PU = {0: [0,1,4,5], 1: [1,5], 2: [2,3,6,7], 3: [3,7]}
q = X[:, 0:12]*capS; d = X[:, 12:24]*capS; qu = X[:, 24:32]*capU; du = X[:, 32:40]*capU
g = tools.greedy(w, X)
m = ok & (c == 0)
np.set_printoptions(precision=2, suppress=True, linewidth=150)
for p in range(4):
    mm = m & (ph == p)
    print(f'phase {p}: n={mm.sum()} hold={(g[mm]==p).mean():.3f}')
    print('   stopline q of cur phase lanes  mean', q[mm][:, PL[p]].mean(0), ' d', d[mm][:, PL[p]].mean(0))
    print('   upstream q of feeding lanes mean', qu[mm][:, PU[p]].mean(0), ' du', du[mm][:, PU[p]].mean(0))
    # quantiles of upstream queue sum
    s = qu[mm][:, PU[p]].sum(1)
    print('   upstream q sum quantiles 50/75/90/99', np.percentile(s, [50,75,90,99]))
    print('   next picks', np.bincount(g[mm], minlength=4))
# per-lane upstream queue overall distribution at ok decisions
print('upstream q per lane, 90th pct:', np.percentile(qu[m], 90, axis=0))
print('upstream q per lane, 99th pct:', np.percentile(qu[m], 99, axis=0))
print('upstream q per lane, max:', qu[m].max(0))
print('upstream d per lane, mean:', du[m].mean(0))
