import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
X = S['obs'].astype(np.float64); ctrl = S['controller']; ep = S['episode']; t = S['time_s']
a = tools.greedy(w, X)
N = tools.names(); ix = {n:i for i,n in enumerate(N)}
SL = {0:['N0','N1','S0','S1'], 1:['N2','S2'], 2:['E0','E1','W0','W1'], 3:['E2','W2']}
UP = {0:['N0u','N1u','S0u','S1u'], 1:['N1u','S1u'], 2:['E0u','E1u','W0u','W1u'], 3:['E1u','W1u']}
def D(X,p,up=True):
    v = sum(X[:, ix['d_'+l]] for l in SL[p])
    if up: v = v + sum(X[:, ix['d_'+l]] for l in UP[p])
    return v
def Q(X,p): return sum(X[:, ix['q_'+l]] for l in SL[p])
ph = X[:,40:44].argmax(1); ok = X[:,44]>0.5
Dsl = np.stack([D(X,p,False) for p in range(4)],1)
Dall = np.stack([D(X,p,True) for p in range(4)],1)
Qs = np.stack([Q(X,p) for p in range(4)],1)
for c in range(3):
    m = (ctrl==c)&ok&(a!=ph)
    chosen_sl = Dsl[m, a[m]]; chosen_q = Qs[m, a[m]]
    others = Dsl[m].copy(); others[np.arange(m.sum()), a[m]] = 0
    print('ctrl',c,'switches',m.sum(),'chosen phase stopline empty:', (chosen_sl==0).sum(),
          ' of which some other phase had stopline vehicles:', ((chosen_sl==0)&(others.max(1)>0)).sum(),
          ' chosen has no queue but another has queue:', ((chosen_q==0)&(np.delete(Qs[m],0,1).max(1)>=0)& (np.array([Qs[i][[p for p in range(4) if p!=ph[i]]].max() for i in np.where(m)[0]])>0)).sum())
    # transition matrix
    T = np.zeros((4,4),int)
    for i in np.where(m)[0]: T[ph[i], a[i]] += 1
    print(T)
