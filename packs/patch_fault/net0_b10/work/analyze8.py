import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
S = tools.states(); X = S['obs'].astype(np.float64); C = S['controller']; T = S['time_s']
cur = X[:, 40:44].argmax(1)
for c in range(3):
    m = (C == c) & (T < 3600)
    qE2, qE1u, qW1u, qE1, qW1 = X[:, 5], X[:, 27], X[:, 31], X[:, 4], X[:, 10]
    a = m & (qE1u > 0.05)
    b = m & (qW1u > 0.05)
    print(S['controller_names'][c],
          f'P(qE2 full | qE1u>0.05)={((qE2[a] > 0.99)).mean():.2f} (n={a.sum()})',
          f'P(qE2 full | qW1u>0.05)={((qE2[b] > 0.99)).mean():.2f} (n={b.sum()})',
          f'P(qE2 full)={((qE2[m] > 0.99)).mean():.2f}',
          f'P(qE2=0 | qW1u>0.05)={((qE2[b] < 0.01)).mean():.2f}',
          f'mean qW1u-qE1u={(qW1u[m]-qE1u[m]).mean():+.3f}')
# for the start network (failed) and fixed-time (W-left served regardless), compare W1u vs E1u queue in phase 2 green
for c in range(3):
    m = (C == c) & (T < 3600) & (cur == 2)
    print(S['controller_names'][c], 'during phase 2: mean q_E1u', X[m, 27].mean().round(3), 'q_W1u', X[m, 31].mean().round(3),
          '| P(q>0.05) E1u', (X[m, 27] > 0.05).mean().round(3), 'W1u', (X[m, 31] > 0.05).mean().round(3))
