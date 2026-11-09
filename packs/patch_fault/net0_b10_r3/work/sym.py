import sys; sys.path.insert(0, '.')
import tools, numpy as np
S = tools.states(); X = S['obs'].astype(np.float64); C = S['controller']
n = tools.names()
pairs = [(3,9),(4,10),(5,11),(15,21),(16,22),(17,23),(26,30),(27,31),(34,38),(35,39),(0,6),(1,7),(2,8)]
for c in range(3):
    m = C==c
    print(S['controller_names'][c], "eps", np.unique(S['episode'][m]))
    print("  ", "  ".join(f"{n[a]}={X[m,a].mean():.3f}/{n[b]}={X[m,b].mean():.3f}" for a,b in pairs))
    print("  corr q_W1u,q_E1u", np.corrcoef(X[m,27],X[m,31])[0,1].round(3), " corr q_E1, q_W1", np.corrcoef(X[m,4],X[m,10])[0,1].round(3),
          " corr q_N2,q_S2", np.corrcoef(X[m,2],X[m,8])[0,1].round(3), " corr d_N2,d_S2", np.corrcoef(X[m,14],X[m,20])[0,1].round(3))
    print("  phase dist", np.bincount(X[m,40:44].argmax(1), minlength=4)/m.sum())
