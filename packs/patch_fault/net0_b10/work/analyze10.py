import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
S = tools.states(); X = S['obs'].astype(np.float64); C = S['controller']; T = S['time_s']
for c in range(3):
    m = (C == c) & (T < 3600)
    print(S['controller_names'][c], 'corr q_N2,q_S2 %.2f  d_N2,d_S2 %.2f | corr q_E2,q_N2 %.2f | means q_E2 %.2f d_E2 %.2f q_N2 %.2f q_S2 %.2f' % (
        np.corrcoef(X[m, 2], X[m, 8])[0, 1], np.corrcoef(X[m, 14], X[m, 20])[0, 1], np.corrcoef(X[m, 5], X[m, 2])[0, 1],
        X[m, 5].mean(), X[m, 17].mean(), X[m, 2].mean(), X[m, 8].mean()))
    # conditional mean of S2 given N2 level (the analogue of W2 given E2)
    print('    E[q_S2 | q_N2 = 0,1,2,3 cars]', [round(X[m & (np.round(X[:, 2] * 2.9867) == k), 8].mean(), 2) for k in range(4)])
