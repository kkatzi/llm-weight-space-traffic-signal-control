"""Choose which hidden units to repurpose: greedy, by the joint mean-ablation decision-change rate on recorded
honoured decisions (a statistic of the existing network, not a weight fit)."""
import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); ok = X[:, 44] > 0.5
W1, b1, W2, b2, W3, b3 = (w[k] for k in ['W1','b1','W2','b2','W3','b3'])
h1 = np.tanh(X @ W1.T + b1); h2 = np.tanh(h1 @ W2.T + b2)
g = (h2 @ W3.T + b3).argmax(1)
m1 = h1.mean(0); m2 = h2.mean(0)
def rate(H1set, H2set):
    hh = h1.copy(); hh[:, H1set] = m1[H1set]
    h2n = np.tanh(hh @ W2.T + b2); h2n[:, H2set] = m2[H2set]
    return (((h2n @ W3.T + b3).argmax(1) != g)[ok]).mean()
# greedy h1 selection (with no h2 ablated)
H1 = []
for _ in range(4):
    best = min((rate(H1 + [j], []), j) for j in range(64) if j not in H1)
    H1.append(best[1]); print('h1 add', best[1], 'rate', round(best[0], 4))
H2 = []
for _ in range(4):
    best = min((rate(H1, H2 + [k]), k) for k in range(64) if k not in H2)
    H2.append(best[1]); print('h2 add', best[1], 'rate', round(best[0], 4))
print('H1', H1, 'H2', H2, 'joint rate', rate(H1, H2))
