import sys; sys.path.insert(0, '.')
import tools, numpy as np
S = tools.states(); X = S['obs'].astype(np.float64); C=S['controller']; E=S['episode']; T=S['time_s']; A=S['act']
n = tools.names()
for c in range(3):
    for e in np.unique(E[C==c]):
        m=(C==c)&(E==e)
        idx=np.where(m)[0]
        last=idx[-1]
        nz={n[i]:round(float(X[last,i]),2) for i in range(40) if X[last,i]>0}
        print(S['controller_names'][c], e, "n",m.sum(), "tmax", T[m].max(), "last phase", X[last,40:44].argmax(), "tg", round(X[last,45],2), nz)
