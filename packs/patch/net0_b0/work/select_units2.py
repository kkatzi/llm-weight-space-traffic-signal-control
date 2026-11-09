from common import *
w = tools.load("../start_weights.npz")
S = tools.states(); X = S['obs'].astype(np.float64); ok = X[:, 44] > 0.5
h1, h2 = tools.hidden(w, X); g0 = tools.greedy(w, X)
m1 = h1.mean(0); m2 = h2.mean(0)
def logits_fixed(h1fix={}, h2fix={}):
    H1 = h1.copy()
    for i, v in h1fix.items(): H1[:, i] = v
    H2 = np.tanh(H1 @ w['W2'].T + w['b2'])
    for j, v in h2fix.items(): H2[:, j] = v
    return H2 @ w['W3'].T + w['b3']
r1 = sorted([((logits_fixed({i: m1[i]}).argmax(1) == g0)[ok].mean(), i) for i in range(64)], reverse=True)
r2 = sorted([((logits_fixed({}, {j: m2[j]}).argmax(1) == g0)[ok].mean(), j) for j in range(64)], reverse=True)
print('layer1 (mean-replace agreement on ok states):', [(round(a, 4), i) for a, i in r1[:12]])
print('layer2:', [(round(a, 4), j) for a, j in r2[:12]])
# greedy pick 4 + 4 jointly (choice of which units to sacrifice, by agreement statistic)
c1 = []; c2 = []
for _ in range(4):
    best = max((i for i in range(64) if i not in c1), key=lambda i: (logits_fixed({k: m1[k] for k in c1 + [i]}).argmax(1) == g0)[ok].mean())
    c1.append(best)
for _ in range(4):
    best = max((j for j in range(64) if j not in c2), key=lambda j: (logits_fixed({k: m1[k] for k in c1}, {k: m2[k] for k in c2 + [j]}).argmax(1) == g0)[ok].mean())
    c2.append(best)
lg = logits_fixed({k: m1[k] for k in c1}, {k: m2[k] for k in c2})
print('chosen layer1', c1, 'layer2', c2, 'agreement ok', (lg.argmax(1) == g0)[ok].mean(), 'all', (lg.argmax(1) == g0).mean())
