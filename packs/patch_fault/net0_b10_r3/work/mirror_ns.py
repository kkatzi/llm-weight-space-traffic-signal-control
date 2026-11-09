"""Stand-in experiment on the fully observed N2/S2 pair: how close does the network's decision stay when the
S2 inputs (x8 q_S2, x20 d_S2) are replaced by various imputations?  (W2 stays dead, as deployed.)"""
import sys; sys.path.insert(0, '.')
import tools, numpy as np
S = tools.states(); X = S['obs'].astype(np.float64); C=S['controller']; T=S['time_s']
w = tools.load("start_weights.npz")
ok = X[:,44]>0.5
ref = tools.greedy(w, X)
def agree(Xm, m):
    return (tools.greedy(w,Xm)[m]==ref[m]).mean()
mq, md = X[:,8].mean(), X[:,20].mean()
for c in [0,1,2]:
    m = ok&(C==c)&(T<3600)
    res={}
    Xm=X.copy(); Xm[:,8]=0; Xm[:,20]=0; res['zero']=agree(Xm,m)
    Xm=X.copy(); Xm[:,8]=X[:,2]; Xm[:,20]=X[:,14]; res['mirror']=agree(Xm,m)
    for a in [0.5,0.75]:
        Xm=X.copy(); Xm[:,8]=a*X[:,2]+(1-a)*mq; Xm[:,20]=a*X[:,14]+(1-a)*md; res[f'shrink{a}']=agree(Xm,m)
    Xm=X.copy(); Xm[:,8]=mq; Xm[:,20]=md; res['mean']=agree(Xm,m)
    print(S['controller_names'][c], {k:round(v,3) for k,v in res.items()}, "n",m.sum())
