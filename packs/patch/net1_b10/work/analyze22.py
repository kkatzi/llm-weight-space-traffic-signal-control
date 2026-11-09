import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); c = S['controller']; t = S['time_s']; ep = S['episode']
ph = X[:, 40:44].argmax(1)
tg = np.round(X[:, 45]*60)
for ci in range(3):
    m = c == ci
    vals, cnt = np.unique(tg[m], return_counts=True)
    print(S['controller_names'][ci], dict(zip(vals.astype(int), cnt)))
m = (c == 0) & (tg >= 27)
for i in np.where(m)[0]:
    print('  t', t[i], 'ep', ep[i], 'ph', ph[i], 'tg', tg[i], 'sum q', round(X[i, :12].sum()*3, 1), 'sum d', round(X[i, 12:24].sum()*3, 1), 'up d', round(X[i, 32:40].sum()*34.1, 1))
