import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
S=tools.states(); X=S['obs'].astype(np.float64); C=S['controller']; ep=S['episode']; T=S['time_s']
names=tools.names()
for c in range(3):
    m=(C==c)
    ud=X[m,32:40]
    print(S['controller_names'][c],'upstream density p99 per lane',np.percentile(ud,99,axis=0).round(2),'max',ud.max(0).round(2))
    print('   stop-line full (d>=0.99) share per lane', (X[m,12:24]>0.99).mean(0).round(2))
