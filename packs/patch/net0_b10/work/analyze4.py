import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
X = S['obs'].astype(np.float64); ctrl = S['controller']; ep = S['episode']; t = S['time_s']
a = tools.greedy(w, X)
N = tools.names(); ix = {n:i for i,n in enumerate(N)}
# lanes served by each phase (stop-line) and upstream feeders
SL = {0:['N0','N1','S0','S1'], 1:['N2','S2'], 2:['E0','E1','W0','W1'], 3:['E2','W2']}
UP = {0:['N0u','N1u','S0u','S1u'], 1:['N1u','S1u'], 2:['E0u','E1u','W0u','W1u'], 3:['E1u','W1u']}
def dem(X, p, up=True):
    v = sum(X[:, ix['d_'+l]] for l in SL[p])
    if up: v = v + sum(X[:, ix['d_'+l]] for l in UP[p])
    return v
def qsl(X,p): return sum(X[:, ix['q_'+l]] for l in SL[p])
ph = X[:,40:44].argmax(1); ok = X[:,44]>0.5; tg = X[:,45]
cur_sl = np.array([dem(X[i:i+1], ph[i], up=False)[0] for i in range(len(X))])
cur_all = np.array([dem(X[i:i+1], ph[i], up=True)[0] for i in range(len(X))])
other_q = np.array([sum(qsl(X[i:i+1],p)[0] for p in range(4) if p!=ph[i]) for i in range(len(X))])
stay = a==ph
for c in range(3):
    m = ctrl==c
    print('ctrl',c,'n',m.sum(),'ok',(m&ok).sum())
    mo = m&ok
    print('  P(net stays | ok)', stay[mo].mean().round(3))
    e1 = mo & (cur_sl==0) & (other_q>0)
    e2 = mo & (cur_all==0) & (other_q>0)
    print('  cur stopline empty & others queued:', e1.sum(), 'net stays', stay[e1].sum())
    print('  cur all empty & others queued:', e2.sum(), 'net stays', stay[e2].sum())
    e3 = mo & (cur_sl>0)
    print('  cur stopline nonempty:', e3.sum(), 'net stays', stay[e3].sum())
    # t_green distribution when ok
    print('  tg hist (ok):', np.histogram(tg[mo]*60, bins=[0,12,13,18,23,30,45,60.1])[0])
# for net's own states: list stays with t_green >= 17s
m = (ctrl==0)&ok&stay
print('net stays on own states:', m.sum())
for i in np.where(m)[0][:40]:
    print(ep[i], t[i], 'ph', ph[i], 'tg %.0f'%(tg[i]*60), 'cur_sl %.2f cur_all %.2f other_q %.2f'%(cur_sl[i], cur_all[i], other_q[i]), tools.forward(w, X[i])[0].round(1))
