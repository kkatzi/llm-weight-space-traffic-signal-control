import sys; sys.path.insert(0, '.')
import tools, numpy as np
S = tools.states(); X = S['obs'].astype(np.float64); C=S['controller']; E=S['episode']; T=S['time_s']
n=tools.names()
ph = X[:,40:44].argmax(1)
m = (C==0)
for e in range(6):
    me = m&(E==e)
    long2 = me&(ph==2)&(X[:,45]>=0.99)
    print("ep",e,"phase2 tg=1 decisions: in hour", (long2&(T<3600)).sum(), "after", (long2&(T>=3600)).sum(),
          " first time tg=1 after 3600:", T[long2&(T>=3600)].min() if (long2&(T>=3600)).any() else None)
# ep5 timeline after 3500
me = m&(E==5)&(T>=3550)&(T<=3800)
for i in np.where(me)[0][::2]:
    nz={n[j]:round(float(X[i,j]),2) for j in range(40) if X[i,j]>0}
    print(T[i], ph[i], round(X[i,45],2), S['act'][i], nz)
