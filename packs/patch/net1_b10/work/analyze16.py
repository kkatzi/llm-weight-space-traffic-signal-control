import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); c = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
L = tools.forward(w, X); g = L.argmax(1)
TL = np.array([1, -1, 1, -1.]); EW = np.array([-1, -1, 1, 1.])
m = ok & (c == 0)
for mag in [0.5, 0.75, 1.0]:
    for sa in [1, -1]:
        for sb in [1, -1]:
            v = mag*(sa*TL + sb*EW)
            g2 = (L + v).argmax(1)
            ch = m & (g2 != g)
            cnt = np.zeros((4,4), int)
            for a, b in zip(g[ch], g2[ch]): cnt[a, b] += 1
            print(f'mag {mag} thr{"+" if sa>0 else "-"} EW{"+" if sb>0 else "-"} b3 delta {v}: changed {ch.mean()/m.mean():.3f}  new holds {(ch & (g2==ph)).sum()} lost holds {(ch & (g==ph)).sum()}')
