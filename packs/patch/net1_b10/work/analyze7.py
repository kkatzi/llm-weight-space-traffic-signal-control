import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); A = S['act']; ep = S['episode']; t = S['time_s']; c = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
capS = 22.4/7.5; capU = 256/7.5
L = tools.forward(w, X)
g = L.argmax(1)
srt = np.sort(L, 1); gap = srt[:, -1] - srt[:, -2]
second = np.argsort(L, 1)[:, -2]
np.set_printoptions(precision=2, suppress=True, linewidth=160)
m = ok
print('ok-state logit gap pct 10/25/50/75', np.percentile(gap[m], [10,25,50,75]))
print('non-ok-state logit gap pct', np.percentile(gap[~m], [10,25,50,75]))
# logits for current phase vs best at ok
cur = L[np.arange(len(X)), ph]
print('ok: best - current logit pct', np.percentile((srt[:, -1]-cur)[m], [5,10,25,50]))
# does the network's pick depend on ok? compare picks with ok flipped
X2 = X.copy(); X2[:, 44] = 1 - X2[:, 44]
g2 = tools.greedy(w, X2)
print('non-ok states: pick==current', (g[~m]==ph[~m]).mean(), ' with ok set to 1, pick==current', (g2[~m]==ph[~m]).mean())
# per t_green bin, hold prob
tg = np.round(X[:, 45]*60)
for v in np.unique(tg)[:12]:
    mm = tg==v
    print(f'  t_green {v:.0f}: n={mm.sum()} ok={ok[mm].mean():.2f} hold={(g[mm]==ph[mm]).mean():.2f}')
