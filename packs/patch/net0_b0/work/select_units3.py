from common import *
exec(open('explore10.py').read().split('X = rand_sparse')[0].split('w = tools.load')[1].split('\n', 1)[1])
w0 = tools.load("../start_weights.npz")
S = tools.states(); X = S['obs'].astype(np.float64); ok = X[:, 44] > 0.5; t = S['time_s']
Xs = rand_sparse(50000); Es = phase_empty(Xs)
a0s = tools.greedy(w0, Xs); keep = ~(Es[np.arange(len(Xs)), a0s] & (~Es).any(1))
Xs = Xs[keep]
sets = {'rec<3600': X[ok & (t < 3600)], 'rec>=3600': X[ok & (t >= 3600)], 'sparse': Xs}
m1 = tools.hidden(w0, X)[0].mean(0); m2 = tools.hidden(w0, X)[1].mean(0)
H = {k: tools.hidden(w0, v) for k, v in sets.items()}
G = {k: tools.greedy(w0, v) for k, v in sets.items()}
def agree(c1, c2, k):
    h1, h2 = H[k]; H1 = h1.copy(); H1[:, c1] = m1[c1]
    H2 = np.tanh(H1 @ w0['W2'].T + w0['b2']); H2[:, c2] = m2[c2]
    return ((H2 @ w0['W3'].T + w0['b3']).argmax(1) == G[k]).mean()
def score(c1, c2):  # worst-case style: weight the real-traffic set most, but count sparse states too
    return 0.6 * agree(c1, c2, 'rec<3600') + 0.2 * agree(c1, c2, 'rec>=3600') + 0.2 * agree(c1, c2, 'sparse')
print('single layer-1 units (rec<3600, rec>=3600, sparse):')
r = sorted(((score([i], []), i) for i in range(64)), reverse=True)
for s, i in r[:10]: print('  %2d %.4f %.4f %.4f' % (i, agree([i], [], 'rec<3600'), agree([i], [], 'rec>=3600'), agree([i], [], 'sparse')))
print('single layer-2 units:')
r = sorted(((score([], [j]), j) for j in range(64)), reverse=True)
for s, j in r[:10]: print('  %2d %.4f %.4f %.4f' % (j, agree([], [j], 'rec<3600'), agree([], [j], 'rec>=3600'), agree([], [j], 'sparse')))
c1 = []; c2 = []
for _ in range(3): c1.append(max((i for i in range(64) if i not in c1), key=lambda i: score(c1 + [i], [])))
for _ in range(3): c2.append(max((j for j in range(64) if j not in c2), key=lambda j: score(c1, c2 + [j])))
print('greedy choice', c1, c2, {k: round(agree(c1, c2, k), 4) for k in sets})
old = ([60, 32, 57], [61, 48, 0]); print('previous choice', old, {k: round(agree(*old, k), 4) for k in sets})
