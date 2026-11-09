import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); c = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
L = tools.forward(w, X); g = L.argmax(1)
tilts = {'thr-vs-left': np.array([1, -1, 1, -1.]), 'EW-vs-NS': np.array([-1, -1, 1, 1.]),
         'EWthr': np.array([0, 0, 1, 0.]), 'EWleft': np.array([0, 0, 0, 1.]), 'NSthr': np.array([1, 0, 0, 0.]), 'NSleft': np.array([0, 1, 0, 0.])}
m = ok & (c == 0)
for name, v in tilts.items():
    for dl in [-1.5, -0.75, 0.75, 1.5]:
        g2 = (L + dl*v).argmax(1)
        ch = m & (g2 != g)
        newhold = ch & (g2 == ph)
        print(f'{name:12s} delta {dl:+.2f}: changed {ch.sum():4d}/{m.sum()} ({ch.mean()/m.mean():.3f}) of which new holds {newhold.sum()}, lost holds {(ch & (g==ph)).sum()}')
