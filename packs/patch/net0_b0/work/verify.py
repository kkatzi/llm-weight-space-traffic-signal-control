from common import *
import json
from collections import Counter
np.set_printoptions(precision=2, suppress=True, linewidth=220)
info = json.load(open('patch_info.json'))
VETO, c1, c2 = info['veto'], info['layer1'], info['layer2']
w0 = tools.load("../start_weights.npz"); w = tools.load("weights.npz")
print('1) edit size', tools.edit_size(w, w0))
S = tools.states(); X = S['obs'].astype(np.float64); ok = X[:, 44] > 0.5; ctrl = S['controller']; t = S['time_s']; ep = S['episode']
g0 = tools.greedy(w0, X); g1 = tools.greedy(w, X)
E = phase_empty(X)
print('2) agreement with original on recorded states: honoured %.4f  all %.4f  (n honoured %d)' % ((g0 == g1)[ok].mean(), (g0 == g1).mean(), ok.sum()))
for c in range(3):
    m = ok & (ctrl == c)
    print('   %-13s honoured agreement %.4f, changed %d' % (S['controller_names'][c], (g0 == g1)[m].mean(), (g0 != g1)[m].sum()))
d = ok & (g0 != g1)
vetoed = d & E[np.arange(len(X)), g0] & (~E).any(1)
print('   changed honoured decisions: %d, of which original picked an empty phase (veto): %d, collateral: %d' % (d.sum(), vetoed.sum(), (d & ~vetoed).sum()))
coll = d & ~vetoed
print('   collateral (orig->new):', Counter(zip(g0[coll].tolist(), g1[coll].tolist())), ' t>=3600:', (coll & (t >= 3600)).sum())
lg0 = tools.forward(w0, X); srt = np.sort(lg0, 1); gap = srt[:, -1] - srt[:, -2]
print('   original top-2 logit gap at collateral states: median %.2f max %.2f (all honoured median %.2f)' % (np.median(gap[coll]), gap[coll].max(), np.median(gap[ok])))
# 3) detector/gate saturation
def acts(w, X):
    a1 = X @ w['W1'].T + w['b1']; H1 = np.tanh(a1); a2 = H1 @ w['W2'].T + w['b2']; return a1, H1, a2, np.tanh(a2)
rng = np.random.default_rng(1)
Xr = rng.random((100000, 46))
Xr[:, 40:44] = np.eye(4)[rng.integers(0, 4, len(Xr))]
Xr[:, 44] = rng.integers(0, 2, len(Xr))
for name, XX in [('recorded', X), ('uniform random', Xr), ('all ones', np.ones((1, 46))), ('all zeros', np.zeros((1, 46)))]:
    a1, H1, a2, H2 = acts(w, XX); Ex = phase_empty(XX)
    worst1 = min(np.abs(a1[:, i]).min() for i in c1); worst2 = min(np.abs(a2[:, j]).min() for j in c2)
    okz = all(np.array_equal(np.sign(a1[:, i]) > 0, Ex[:, q]) for q, i in zip(VETO, c1))
    okg = all(np.array_equal(np.sign(a2[:, j]) > 0, Ex[:, q]) for q, j in zip(VETO, c2))
    print('3) %-15s min|pre| detectors %.1f gates %.1f, detectors match emptiness: %s, gates match: %s' % (name, worst1, worst2, okz, okg))
# 4) the patched network equals: original net with sacrificed units at their mean, minus V on empty phases
h1, h2 = tools.hidden(w0, np.vstack([X, Xr]))
XX = np.vstack([X, Xr])
m1 = tools.hidden(w0, X)[0].mean(0); m2 = tools.hidden(w0, X)[1].mean(0)
H1 = h1.copy(); H1[:, c1] = m1[c1]; H2 = np.tanh(H1 @ w0['W2'].T + w0['b2']); H2[:, c2] = m2[c2]
lgv = H2 @ w0['W3'].T + w0['b3']; EE = phase_empty(XX)
for q in VETO: lgv[:, q] -= info['V'] * EE[:, q]
lgp = tools.forward(w, XX)
print('4) max |patched logits - (mean-replaced original - V*empty)| = %.2e' % np.abs(lgp - lgv).max())
# 5) static end-of-episode traces with the real patched network
pol = lambda x: tools.greedy(w, x)
fails, _ = all_traces(pol)
print('5) static sparse traces (4095 lane sets x 3 counts x 4 start phases): failures', len(fails), fails[:5])
# 6) random sparse states
exec(open('explore10.py').read().split('X = rand_sparse')[0].split('w = tools.load')[1].split('\n', 1)[1])
Xs = rand_sparse(200000); Es = phase_empty(Xs); a = tools.greedy(w, Xs)
bad = (~Es).any(1) & Es[np.arange(len(Xs)), a]
print('6) random sparse states: picks an empty phase while another is loaded:', bad.sum(), 'of', len(Xs))
a0 = tools.greedy(w0, Xs); mod = Es[np.arange(len(Xs)), a0] & (~Es).any(1)
print('   agreement with original where original picked a loaded phase: %.4f' % (a[~mod] == a0[~mod]).mean())
# 7) the recorded deadlock
m = (ctrl == 0) & (ep == 2) & (t == 4000); x = X[m][0]
print('7) recorded deadlock state: original logits', tools.forward(w0, x)[0], '-> patched', tools.forward(w, x)[0], 'pick', tools.greedy(w, x)[0])
r, path = trace(pol, [0, 1, 11], start=2); print('   trace N0+N1+W2 waiting, EW green:', r, path)
r, path = trace(pol, [0, 1], start=3); print('   trace N0+N1 waiting, EW-left green:', r, path)
print('8) trainability original', tools.trainability(w0))
print('   trainability patched ', tools.trainability(w))
