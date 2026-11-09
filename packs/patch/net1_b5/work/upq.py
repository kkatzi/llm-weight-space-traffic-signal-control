from common import *
g=tools.greedy(W0,X); P=tools.probs(W0,X)
ups=['N0u','N1u','E0u','E1u','S0u','S1u','W0u','W1u']
serve={'N0u':[0],'N1u':[0,1],'E0u':[2],'E1u':[2,3],'S0u':[0],'S1u':[0,1],'W0u':[2],'W1u':[2,3]}
for k,u in enumerate(ups):
    for thr in [0.1,0.25]:
        m=ok&(X[:,24+k]>thr)
        if m.sum()==0: continue
        T=np.bincount(g[m],minlength=4)
        print(f'{u} q>{thr} n={m.sum():4d} picks {T} cur {np.bincount(cur[m],minlength=4)} served-by {serve[u]}')
