from common import *
import json
exec(open('explore10.py').read().split('X = rand_sparse')[0].split('w = tools.load')[1].split('\n', 1)[1])
info = json.load(open('patch_info.json')); c1, c2, V = info['layer1'], info['layer2'], info['V']
w0 = tools.load("../start_weights.npz"); w = tools.load("weights.npz")
S = tools.states(); X = S['obs'].astype(np.float64)
m1 = tools.hidden(w0, X)[0].mean(0); m2 = tools.hidden(w0, X)[1].mean(0)
Xs = rand_sparse(50000); Es = phase_empty(Xs)
h1, h2 = tools.hidden(w0, Xs); H1 = h1.copy(); H1[:, c1] = m1[c1]; H2 = np.tanh(H1 @ w0['W2'].T + w0['b2']); H2[:, c2] = m2[c2]
lgm = H2 @ w0['W3'].T + w0['b3']; lgv = lgm.copy()
for q in info['veto']: lgv[:, q] -= V * Es[:, q]
lgp = tools.forward(w, Xs); lg0 = tools.forward(w0, Xs)
print('max diff patched vs virtual', np.abs(lgp - lgv).max())
a0 = lg0.argmax(1); am = lgm.argmax(1); ap = lgp.argmax(1)
loadedpick = ~(Es[np.arange(len(Xs)), a0] & (~Es).any(1))
print('agree mean-replaced vs orig (loaded picks)', (am == a0)[loadedpick].mean(), ' patched vs orig', (ap == a0)[loadedpick].mean())
diff = loadedpick & (ap != a0)
print('n', loadedpick.sum(), 'diff', diff.sum(), ' of which mean-replaced also differs', (diff & (am != a0)).sum())
i = np.where(diff & (am == a0))[0][:5]
for k in i: print(np.round(lg0[k], 2), np.round(lgm[k], 2), np.round(lgp[k], 2), Es[k], a0[k], ap[k])
