import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
w=tools.load('start_weights.npz')
S=tools.states()
X=S['obs'].astype(np.float64); A=S['act']; ep=S['episode']; T=S['time_s']; C=S['controller']
names=tools.names()
g=tools.greedy(w,X)
print('greedy==act on start_network states:', (g[C==0]==A[C==0]).mean())
for c in range(3):
    m=C==c
    print(S['controller_names'][c], 'episodes', np.unique(ep[m]), 'n', m.sum())
# per episode length
for c in range(3):
    for e in np.unique(ep[C==c]):
        m=(C==c)&(ep==e)
        print(c,e,'len',m.sum(),'tmax',T[m].max())
