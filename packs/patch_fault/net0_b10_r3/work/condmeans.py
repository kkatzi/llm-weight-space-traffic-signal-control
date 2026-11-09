import sys; sys.path.insert(0, '.')
import tools, numpy as np
S = tools.states(); X = S['obs'].astype(np.float64); C=S['controller']; T=S['time_s']
ph = X[:,40:44].argmax(1)
for c in [1,0,2]:
    m=(C==c)&(T<3600)
    print(S['controller_names'][c])
    for lo,hi in [(-1,0.001),(0.001,0.05),(0.05,0.1),(0.1,0.2),(0.2,1.1)]:
        mm=m&(X[:,27]>lo)&(X[:,27]<=hi)
        mm2=m&(X[:,2+0*0+25-25+25-25]>-9)  # placeholder
        if mm.sum(): print(f"  q_E1u in ({lo},{hi}]: n={mm.sum():5d}  E[q_E2]={X[mm,5].mean():.2f}  P(q_E2>0.9)={(X[mm,5]>0.9).mean():.2f}  E[d_E1]={X[mm,16].mean():.2f}")
    # same for N side: q_N1u vs q_N2
    for lo,hi in [(-1,0.001),(0.001,0.05),(0.05,0.1),(0.1,1.1)]:
        mm=m&(X[:,25]>lo)&(X[:,25]<=hi)
        if mm.sum(): print(f"  q_N1u in ({lo},{hi}]: n={mm.sum():5d}  E[q_N2]={X[mm,2].mean():.2f}  P(q_N2>0.9)={(X[mm,2]>0.9).mean():.2f}")
    # W side under this controller for reference
    for lo,hi in [(-1,0.001),(0.001,0.05),(0.05,0.1),(0.1,0.2),(0.2,1.1)]:
        mm=m&(X[:,31]>lo)&(X[:,31]<=hi)
        if mm.sum(): print(f"  q_W1u in ({lo},{hi}]: n={mm.sum():5d}  E[q_E2]={X[mm,5].mean():.2f}  E[d_W1]={X[mm,22].mean():.2f}")
