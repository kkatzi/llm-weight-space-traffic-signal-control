import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
import verify2
w0 = tools.load('start_weights.npz')
X = tools.states()['obs'].astype(float); ok = X[:, 44] > 0.5
tt = verify2.truth(X); anyrule = np.zeros(len(X), bool)
for v in tt.values(): anyrule |= v
lg0 = tools.forward(w0, X); srt = np.sort(lg0, 1); gap = srt[:, -1] - srt[:, -2]
h1, h2 = tools.hidden(w0, X)
g0 = lg0.argmax(1)
def flips(h1u, h2u):
    H1 = h1.copy(); H1[:, h1u] = h1[:, h1u].mean(0)
    H2 = np.tanh(H1 @ w0['W2'].T + w0['b2'])
    H2[:, h2u] = H2[:, h2u].mean(0)   # approx (mean of new)
    g = (H2 @ w0['W3'].T + w0['b3']).argmax(1)
    m = ok & ~anyrule
    return (g[m] != g0[m]).sum(), g
w = tools.load(sys.argv[1])
g = tools.greedy(w, X)
m = ok & ~anyrule & (g != g0)
print('flipped outside rules:', m.sum(), ' original gap quantiles at flips', np.quantile(gap[m], [.5, .9, 1]).round(2))
print('h1 {50,42,38,5,51} h2 {1,14,52,63}:', flips([50, 42, 38, 5, 51], [1, 14, 52, 63])[0])
print('drop E1 (h1 38, h2 52):', flips([50, 42, 5, 51], [1, 14, 63])[0])
# rank alternatives for each slot greedily
cand1 = [50, 42, 38, 5, 51, 6, 48, 43, 49, 19, 11, 0]
cand2 = [1, 14, 52, 63, 7, 20, 9, 36, 56, 25, 27, 54]
best = None
for h1set in itertools.combinations(cand1[1:], 4):
    f, _ = flips([50] + list(h1set), [1, 14, 52, 63])
    if best is None or f < best[0]: best = (f, h1set)
print('best h1 set (with 50) for fixed gates:', best)
best2 = None
for h2set in itertools.combinations(cand2[1:], 3):
    f, _ = flips([50] + list(best[1]), [1] + list(h2set))
    if best2 is None or f < best2[0]: best2 = (f, h2set)
print('best gate set (with 1):', best2)
