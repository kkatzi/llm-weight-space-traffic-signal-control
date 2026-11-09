from common import *
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w = tools.load("../start_weights.npz")
S = tools.states()
X = S['obs'].astype(np.float64)
ok = X[:, 44] > 0.5
h1, h2 = tools.hidden(w, X)
g0 = tools.greedy(w, X)
m1 = h1.mean(0); s1 = h1.std(0); m2 = h2.mean(0); s2 = h2.std(0)

def fwd_mod(w, X, h1fix=None, h2fix=None):
    a1 = X @ w['W1'].T + w['b1']; H1 = np.tanh(a1)
    if h1fix:
        for i, v in h1fix.items(): H1[:, i] = v
    a2 = H1 @ w['W2'].T + w['b2']; H2 = np.tanh(a2)
    if h2fix:
        for j, v in h2fix.items(): H2[:, j] = v
    return H2 @ w['W3'].T + w['b3']

rows = []
for i in range(64):
    w2 = {k: v.copy() for k, v in w.items()}
    w2['b2'] = w['b2'] + w['W2'][:, i] * (m1[i] + 1)
    lgA = fwd_mod(w2, X, {i: -1.0})          # normal state of a detector
    lgB = fwd_mod(w2, X, {i: +1.0})          # empty state (perturbation)
    agA = (lgA.argmax(1) == g0)[ok].mean(); agB = (lgB.argmax(1) == g0)[ok].mean()
    resid = np.abs(w['W1'][i]).sum() + abs(w['b1'][i])
    rows.append((agA, agB, i, s1[i], np.linalg.norm(w['W2'][:, i]), resid))
rows.sort(reverse=True)
print('layer-1 candidates: agree(-1) agree(+1) unit std |W2col| rowabs')
for r in rows[:16]: print('  %.4f %.4f  %2d  %.3f %.3f %.2f' % r)

rows2 = []
a2 = h1 @ w['W2'].T + w['b2']
for j in range(64):
    w2 = {k: v.copy() for k, v in w.items()}
    w2['b3'] = w['b3'] + w['W3'][:, j] * (m2[j] + 1)
    lg = fwd_mod(w2, X, None, {j: -1.0})
    ag = (lg.argmax(1) == g0)[ok].mean(); agall = (lg.argmax(1) == g0).mean()
    rows2.append((ag, agall, j, s2[j], m2[j], np.abs(a2[:, j]).max(), np.abs(w['W2'][j]).sum() + abs(w['b2'][j])))
rows2.sort(reverse=True)
print('layer-2 candidates: agree_ok agree_all unit std mean max|pre| rowabs')
for r in rows2[:16]: print('  %.4f %.4f  %2d  %.3f %.3f %.2f %.2f' % r)
