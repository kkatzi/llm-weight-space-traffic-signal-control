import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w0 = tools.load('start_weights.npz'); wp = tools.load('work/proxy_a1.npz')
S = tools.states(); X = S['obs'].astype(np.float64); C = S['controller']; T = S['time_s']; E = S['episode']
ok = X[:,44] > 0.5
g = tools.greedy(wp, X)
cur = X[:,40:44].argmax(1)
m = (C==0)
print('net own states: t_green*60 quantiles when ok', np.quantile(X[m&ok,45]*60, [.1,.5,.9,.99]))
print('stay rate when ok (proxy decisions), per current phase:')
for p in range(4):
    mm = ok & (cur==p)
    print(p, mm.sum(), 'stay', (g[mm]==p).mean().round(3), 'choice dist', np.bincount(g[mm], minlength=4)/mm.sum())
h1, h2 = tools.hidden(wp, X)
g_ref = g
res = []
for j in range(64):
    w = {k: v.copy() for k, v in wp.items()}
    mu = h1[:, j].mean()
    w['b2'] = w['b2'] + w['W2'][:, j] * mu
    w['W2'][:, j] = 0
    gg = tools.greedy(w, X)
    res.append(((gg[ok]==g_ref[ok]).mean(), j, mu, h1[:, j].std(), np.abs(h1[:,j]).min()))
res.sort(reverse=True)
print('least important L1 units (agreement when mean-replaced, j, mean, std, min|h|):')
for r in res[:12]: print(np.round(r, 4))
a1 = X @ wp['W1'].T + wp['b1']
sat = [(np.mean(np.abs(a1[:,j])>2.5), j, h1[:,j].mean()) for j in range(64)]
sat.sort(reverse=True); print('most saturated L1', [tuple(np.round(s,3)) for s in sat[:8]])
res2=[]
for k in range(64):
    w = {kk: v.copy() for kk, v in wp.items()}
    mu = h2[:, k].mean()
    w['b3'] = w['b3'] + w['W3'][:, k] * mu
    w['W3'][:, k] = 0
    gg = tools.greedy(w, X)
    res2.append(((gg[ok]==g_ref[ok]).mean(), k, mu, h2[:, k].std(), w['W3'][:,k].tolist() if False else 0))
res2.sort(reverse=True)
print('least important L2 units:')
for r in res2[:10]: print(np.round(r[:4], 4), 'W3col', wp['W3'][:, r[1]].round(2))
