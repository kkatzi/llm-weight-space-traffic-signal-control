import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load('start_weights.npz')
S = tools.states()
X = S['obs'].astype(float); C = S['controller']; T = S['time_s']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
lg = tools.forward(w, X); g = lg.argmax(1)
q = np.round(X[:, :12] * 3); d = np.round(X[:, 12:24] * 3)
Q = np.stack([q[:, SERVE[p]].sum(1) for p in range(4)], 1)
srt = np.sort(lg, 1); gap = srt[:, -1] - srt[:, -2]
sec = np.argsort(lg, 1)[:, -2]
for thr in [0.1, 0.3, 0.5, 1.0]:
    m = ok & (gap < thr)
    pairs = {}
    for a, b in zip(g[m], sec[m]):
        k = tuple(sorted((a, b))); pairs[k] = pairs.get(k, 0) + 1
    print('gap <', thr, 'n', m.sum(), 'of', ok.sum(), 'pairs', dict(sorted(pairs.items(), key=lambda kv: -kv[1])))
# ties 0 vs 3 from phase 2
m = ok & (ph == 2) & (gap < 0.3) & (((g == 0) & (sec == 3)) | ((g == 3) & (sec == 0)))
print('phase2 0-vs-3 ties', m.sum(), 'picks 0:', (g[m] == 0).sum(), 'picks 3:', (g[m] == 3).sum())
print(' Q0 > Q3 share', (Q[m, 0] > Q[m, 3]).mean().round(2), 'mean Q0', Q[m, 0].mean().round(2), 'mean Q3', Q[m, 3].mean().round(2))
print(' when picks 0: mean Q0 %.2f Q3 %.2f ; when picks 3: mean Q0 %.2f Q3 %.2f' % (Q[m & (g == 0), 0].mean(), Q[m & (g == 0), 3].mean(), Q[m & (g == 3), 0].mean(), Q[m & (g == 3), 3].mean()))
idx = np.where(m)[0]
for i in idx[:25]:
    print(C[i], T[i], 'pick', g[i], 'lg', lg[i].round(2), 'Q', Q[i], 'q', q[i], 'uq', (X[i, 24:32]*34.13).round(0))
# the network's h2 saturation in these states
h1, h2 = tools.hidden(w, X)
print('mean frac saturated h2 (|h|>0.99) in ties vs all ok:', (np.abs(h2[m]) > 0.99).mean().round(3), (np.abs(h2[ok]) > 0.99).mean().round(3))
