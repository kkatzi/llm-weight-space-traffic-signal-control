"""Build work/weights.npz: an 'empty-phase veto' circuit patched into the start network.

For each phase q in 0..3:
  * one layer-1 unit i_q becomes a detector Z_q = +1 if phase q has no vehicle on any lane that feeds it
    (the stop-line lanes it serves + their upstream lanes), else -1.  Row: -K on those density inputs, bias K*theta
    (theta = half an upstream vehicle); K is so large that the untouched row entries cannot move the sign.
  * its old outgoing W2 column is zeroed (its old mean contribution moved into b2) except one entry B into a
    layer-2 unit j_q, which then simply copies Z_q (saturated at +-1).
  * the W3 column of j_q is zero except -V/2 on logit q; b3 absorbs the constant (+V/2) and the old mean
    contribution of j_q.  Net effect: logit_q -= V exactly when phase q is empty, nothing otherwise.
    All four phases are covered so that an all-empty junction shifts every logit equally (original choice kept).
Which units to sacrifice is chosen by a statistic: replacing their activation by its mean must change as few
decisions as possible on the recorded honoured states (weight 0.6 before 3600 s, 0.2 after) and on random sparse
end-of-episode-like states (weight 0.2).
"""
from common import *
import json

VETO = (0, 1, 2, 3)
K = 2000.0                 # detector gain
THETA = 0.5 * ONEU         # half an upstream vehicle
B = 25.0                   # gate weight into the layer-2 unit
V = 40.0                   # logit penalty for an empty phase

exec(open('explore10.py').read().split('X = rand_sparse')[0].split('w = tools.load')[1].split('\n', 1)[1])  # rand_sparse
w0 = tools.load("../start_weights.npz")
S = tools.states(); X = S['obs'].astype(np.float64); ok = X[:, 44] > 0.5; t = S['time_s']
h1all, h2all = tools.hidden(w0, X); m1 = h1all.mean(0); m2 = h2all.mean(0)
Xs = rand_sparse(50000); Es = phase_empty(Xs)
a0s = tools.greedy(w0, Xs); Xs = Xs[~(Es[np.arange(len(Xs)), a0s] & (~Es).any(1))]
sets = {'rec<3600': X[ok & (t < 3600)], 'rec>=3600': X[ok & (t >= 3600)], 'sparse': Xs}
H = {k: tools.hidden(w0, v) for k, v in sets.items()}; G = {k: tools.greedy(w0, v) for k, v in sets.items()}
def agree(c1, c2, k):
    h1, h2 = H[k]; H1 = h1.copy(); H1[:, c1] = m1[c1]
    H2 = np.tanh(H1 @ w0['W2'].T + w0['b2']); H2[:, c2] = m2[c2]
    return ((H2 @ w0['W3'].T + w0['b3']).argmax(1) == G[k]).mean()
def score(c1, c2):
    return 0.6 * agree(c1, c2, 'rec<3600') + 0.2 * agree(c1, c2, 'rec>=3600') + 0.2 * agree(c1, c2, 'sparse')

n = len(VETO); c1 = []; c2 = []
for _ in range(n): c1.append(max((i for i in range(64) if i not in c1), key=lambda i: score(c1 + [i], [])))
for _ in range(n): c2.append(max((j for j in range(64) if j not in c2), key=lambda j: score(c1, c2 + [j])))
print('sacrificed layer-1 units', c1, 'layer-2 units', c2, {k: round(agree(c1, c2, k), 4) for k in sets})

w = {k: v.copy() for k, v in w0.items()}
for i in c1:
    w['b2'] += w0['W2'][:, i] * m1[i]
    w['W2'][:, i] = 0.0
for q, i, j in zip(VETO, c1, c2):
    w['W1'][i, DL[q]] = -K
    w['b1'][i] = K * THETA
    w['W2'][j, i] = B
    w['b3'] += w0['W3'][:, j] * m2[j]
    w['W3'][:, j] = 0.0
    w['W3'][q, j] = -V / 2
    w['b3'][q] -= V / 2
tools.save(w, "weights.npz")
w = tools.load("weights.npz")
print('edit size', tools.edit_size(w, w0))
json.dump({'veto': list(VETO), 'layer1': [int(i) for i in c1], 'layer2': [int(j) for j in c2], 'K': K, 'theta': THETA,
           'B': B, 'V': V}, open('patch_info.json', 'w'), indent=1)
