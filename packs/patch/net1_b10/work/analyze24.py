import numpy as np, sys
sys.path.insert(0, '.')
import tools
W0 = tools.load('start_weights.npz'); S = tools.states(); X = S['obs'].astype(np.float64); c = S['controller']
ok = X[:, 44] > 0.5
W1, b1, W2, b2, W3, b3 = (W0[k] for k in ['W1','b1','W2','b2','W3','b3'])
h1 = np.tanh(X @ W1.T + b1); h2 = np.tanh(h1 @ W2.T + b2); L = h2 @ W3.T + b3; g = L.argmax(1)
def rate(H1set, H2set, mask):
    hh = h1.copy(); hh[:, H1set] = h1[:, H1set].mean(0)
    h2n = np.tanh(hh @ W2.T + b2); h2n[:, H2set] = h2[:, H2set].mean(0)
    return (((h2n @ W3.T + b3).argmax(1) != g)[mask]).mean()
for name, m in [('all ok', ok), ('start-net ok', ok & (c == 0))]:
    print(name, ' h1{31,0}:', round(rate([31, 0], [], m), 4), ' h2{25,15}:', round(rate([], [25, 15], m), 4), ' both:', round(rate([31, 0], [25, 15], m), 4), ' h1{31}:', round(rate([31], [], m), 4))
# close calls: how many ok states have top-2 logit gap < 0.05 (sensitive to any rounding)
srt = np.sort(L, 1); gap = srt[:, -1] - srt[:, -2]
print('ok states with logit gap < 0.05:', (ok & (gap < 0.05)).sum(), ' <0.2:', (ok & (gap < 0.2)).sum(), ' of', ok.sum())
