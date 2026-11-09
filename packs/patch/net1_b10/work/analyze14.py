import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); c = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
nm = tools.names()
L = tools.forward(w, X); g = L.argmax(1); N = len(X)
o = np.argsort(L, 1); sec = o[:, -2]
# gradient of logit difference (pick - second) wrt x, analytic
W1, b1, W2, b2, W3 = w['W1'], w['b1'], w['W2'], w['b2'], w['W3']
a1 = X @ W1.T + b1; h1 = np.tanh(a1); a2 = h1 @ W2.T + b2; h2 = np.tanh(a2)
m = np.where(ok)[0]
G = np.zeros((len(m), 46))
for n, i in enumerate(m):
    v = W3[g[i]] - W3[sec[i]]           # d diff / d h2
    v = v * (1 - h2[i]**2)               # d / d a2
    v = v @ W2                            # d / d h1
    v = v * (1 - h1[i]**2)
    G[n] = v @ W1
# scale by feature spread (std over ok states) to compare importance
sd = X[m].std(0)
imp = np.abs(G).mean(0) * sd
order = np.argsort(-imp)
for k in order:
    print(f'{nm[k]:8s} mean|grad| {np.abs(G[:, k]).mean():6.2f}  sd {sd[k]:.3f}  importance {imp[k]:.3f}')
