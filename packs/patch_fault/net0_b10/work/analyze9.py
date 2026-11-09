import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
S = tools.states(); X = S['obs'].astype(np.float64); C = S['controller']; T = S['time_s']
m = (T < 3600) & (C != 1)   # network + actuated (demand-responsive) states
bins = [0, 0.001, 0.03, 0.06, 0.1, 0.15, 0.25, 1.01]
print('bin of own upstream q        | E[q_E2|q_E1u] E[d_E2|q_E1u] | E[q_E2|q_W1u] E[d_E2|q_W1u] | n_E n_W')
for lo, hi in zip(bins[:-1], bins[1:]):
    a = m & (X[:, 27] >= lo) & (X[:, 27] < hi); b = m & (X[:, 31] >= lo) & (X[:, 31] < hi)
    print(f'[{lo:.3f},{hi:.3f})  {X[a,5].mean():.2f} {X[a,17].mean():.2f} | {X[b,5].mean():.2f} {X[b,17].mean():.2f} | {a.sum()} {b.sum()}')
