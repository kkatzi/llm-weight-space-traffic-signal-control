import sys; sys.path.insert(0, '.')
import tools, numpy as np
S = tools.states(); X = S['obs'].astype(np.float64); C=S['controller']; T=S['time_s']
n=tools.names()
# For E2 (mirror of W2), correlation with each feature, under fixed (symmetric, detector-blind) and start net
for c in [1,0]:
    m=(C==c)&(T<3600)
    for tgt in [5,17]:
        r=[(n[j], np.corrcoef(X[m,tgt],X[m,j])[0,1]) for j in range(40) if j not in (tgt,11,23) and X[m,j].std()>0]
        r.sort(key=lambda t:-abs(t[1]))
        print(S['controller_names'][c], n[tgt], " ".join(f"{a}:{b:.2f}" for a,b in r[:10]))
    # N2 vs S2 check
    print("   corr(qE2, qW1u-mirror qE1u)", round(np.corrcoef(X[m,5],X[m,27])[0,1],3), "corr(q_N2,q_N1u)", round(np.corrcoef(X[m,2],X[m,25])[0,1],3))
