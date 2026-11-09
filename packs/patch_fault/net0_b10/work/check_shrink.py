import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
from make_shrink import build
from check_final import sim, serve
names = tools.names(); capS = 22.4 / 7.5
S = tools.states(); X = S['obs'].astype(np.float64); C = S['controller']; T = S['time_s']
ok = (X[:, 44] > 0.5) & (T < 3600)
gf = tools.greedy(tools.load('work/final_w067_n035.npz'), X)
for rho in [0.3, 0.5, 0.7]:
    w = build(rho, rho, 0.38, 0.47)
    g = tools.greedy(w, X)
    print(f'rho {rho}: edits {tools.edit_size(w, tools.load("start_weights.npz"))["changed"]}')
    for c in range(3):
        m = ok & (C == c)
        print(f'   ctrl {c}: agree with final {np.mean(g[m]==gf[m]):.3f}; phase dist final {np.round(np.bincount(gf[m],minlength=4)/m.sum(),3)} -> {np.round(np.bincount(g[m],minlength=4)/m.sum(),3)}')
    m = ok & (C == 0)
    for e2 in [0, 1, 2, 3]:
        mm = m & (np.round(X[:, 5] * 2.9867) == e2)
        print(f'   net states with E2={e2} cars: phase3 share final {np.mean(gf[mm]==3):.3f} -> {np.mean(g[mm]==3):.3f} (n={mm.sum()})')
    s = sim(w, []); print('   empty idle ->', s[:4])
