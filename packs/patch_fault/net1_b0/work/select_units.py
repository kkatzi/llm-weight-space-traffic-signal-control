"""Pick low-importance hidden units to repurpose (greedy, combined ablation with mean compensation)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
S = tools.states(); X = S['obs'].astype(float)
w = tools.load('start_weights.npz')
Y = X.copy(); Y[:, 11] = Y[:, 5]; Y[:, 23] = Y[:, 17]
ok = X[:, 44] > 0.5
h1, h2 = tools.hidden(w, Y)
base = (h2 @ w['W3'].T + w['b3']).argmax(1)
m1 = h1.mean(0); m2 = h2.mean(0)

def l1_ablate(units):
    H = h1.copy(); H[:, units] = m1[units]
    return (np.tanh(H @ w['W2'].T + w['b2']) @ w['W3'].T + w['b3']).argmax(1)

def l2_ablate(units, H2=None):
    H = (h2 if H2 is None else H2).copy(); H[:, units] = m2[units]
    return (H @ w['W3'].T + w['b3']).argmax(1)

chosen = []
for _ in range(6):
    best = min((u for u in range(64) if u not in chosen), key=lambda u: (l1_ablate(chosen + [u])[ok] != base[ok]).mean())
    chosen.append(best)
    print('L1 set', chosen, 'change %.4f' % (l1_ablate(chosen)[ok] != base[ok]).mean())
ch2 = []
for _ in range(6):
    best = min((u for u in range(64) if u not in ch2), key=lambda u: (l2_ablate(ch2 + [u])[ok] != base[ok]).mean())
    ch2.append(best)
    print('L2 set', ch2, 'change %.4f' % (l2_ablate(ch2)[ok] != base[ok]).mean())
