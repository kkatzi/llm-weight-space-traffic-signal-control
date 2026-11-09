import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
import importlib, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = ['x', 'work/r4.npz']; import verify2
w0 = tools.load('start_weights.npz')
S = tools.states(); X = S['obs'].astype(float); C = S['controller']; T = S['time_s']
ok = X[:, 44] > 0.5
tt = verify2.truth(X); anyrule = np.zeros(len(X), bool)
for v in tt.values(): anyrule |= v
g0 = tools.greedy(w0, X)
for f in ['work/r1.npz', 'work/r4.npz', 'work/r5.npz']:
    g = tools.greedy(tools.load(f), X)
    m = ok & (C == 0)
    print(f, 'start-net trajectories: ok decisions %d, changed in rule states %d, changed elsewhere %d (per episode %.1f), rule changes before t=3600: %d' % (
        m.sum(), (m & anyrule & (g != g0)).sum(), (m & ~anyrule & (g != g0)).sum(), (m & ~anyrule & (g != g0)).sum() / 6,
        (m & anyrule & (g != g0) & (T < 3600)).sum()))
