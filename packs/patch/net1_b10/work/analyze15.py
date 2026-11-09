import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); c = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
h1, h2 = tools.hidden(w, X)
np.set_printoptions(precision=2, suppress=True, linewidth=170)
def phase_r2(H):
    out = []
    for k in range(H.shape[1]):
        y = H[:, k]
        means = np.array([y[ph == p].mean() for p in range(4)])
        res = y - means[ph]
        r2 = 1 - res.var() / y.var()
        out.append((r2, means, y.std()))
    return out
for name, H in [('h1', h1), ('h2', h2)]:
    r = phase_r2(H[ok]) if False else phase_r2(H)
    order = sorted(range(len(r)), key=lambda k: -r[k][0])
    print(name, 'units best explained by current phase:')
    for k in order[:12]:
        print(f'   unit {k:2d} R2 {r[k][0]:.3f} per-phase means {r[k][1]}  std {r[k][2]:.2f}')
