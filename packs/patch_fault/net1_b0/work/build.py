"""Build work/weights.npz from start_weights.npz by construction (no fitting).

1. Mirror proxy for the dead W2 detector: the network sees q_E2 in place of q_W2 and d_E2 in place of d_W2
   (W1[:,5] += W1[:,11]; W1[:,17] += W1[:,23]) on all non-repurposed layer-1 units.
2. Max-green round robin: 4 layer-1 trigger units a_p = [phase p AND t_green + c*P >= 0.975]
   (P = sum of the 8 upstream queues), each driving one layer-2 gate unit that adds +M to logit (p+1)%4.
3. Congestion sequence lock: 4 layer-1 units s_p = [phase p AND P >= P0], each driving a layer-2 gate unit that
   subtracts M from logits (p+2)%4 and (p+3)%4 (only "hold p" or "advance to p+1" remain possible).
Repurposed units were chosen by low ablation impact (select_units.py logic, recomputed here).
b2 / b3 absorb the mean shift so untouched decisions stay as close as possible to the mirror network.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools, numpy as np

# ---- constants (chosen by reasoning + recorded-data quantiles, see NOTES.md)
K_A = 1000.0      # sharpness of the max-green trigger
C_P = 0.0875      # max-green shortening per unit of upstream queue (P=0 -> 60 s, P=6 -> ~27 s)
TH_A = 1.975      # phase-one-hot (1) + tg threshold 0.975
B_C = 40.0        # sharpness of congestion trigger in P
P0 = 1.25         # congestion threshold on P (normal operation max in data: 0.94)
A_C = B_C * (8 - P0) + 60.0
G = 60.0          # layer-2 gate gain
M = 30.0          # logit swing of a gate (off -> on)
UP = list(range(24, 32))

w0 = tools.load(os.path.join(os.path.dirname(HERE), 'start_weights.npz'))
S = tools.states(); X = S['obs'].astype(float)
Ym = X.copy(); Ym[:, 11] = Ym[:, 5]; Ym[:, 23] = Ym[:, 17]
ok = X[:, 44] > 0.5
h1o, h2o = tools.hidden(w0, Ym)
base = (h2o @ w0['W3'].T + w0['b3']).argmax(1)
m1 = h1o.mean(0); m2 = h2o.mean(0)

def greedy_pick(n, ablate):
    ch = []
    for _ in range(n):
        best = min((u for u in range(64) if u not in ch), key=lambda u: (ablate(ch + [u])[ok] != base[ok]).mean())
        ch.append(best)
    return ch

def l1_ablate(units):
    H = h1o.copy(); H[:, units] = m1[units]
    return (np.tanh(H @ w0['W2'].T + w0['b2']) @ w0['W3'].T + w0['b3']).argmax(1)

def l2_ablate(units):
    H = h2o.copy(); H[:, units] = m2[units]
    return (H @ w0['W3'].T + w0['b3']).argmax(1)

L1U = greedy_pick(8, l1_ablate)
L2U = greedy_pick(8, l2_ablate)
A_UNITS, C_UNITS = L1U[:4], L1U[4:]
GA, GC = L2U[:4], L2U[4:]

w = {k: v.copy() for k, v in w0.items()}
# 1. mirror proxy on non-repurposed rows
rows = [i for i in range(64) if i not in L1U]
w['W1'][rows, 5] += w0['W1'][rows, 11]
w['W1'][rows, 17] += w0['W1'][rows, 23]
# 2./3. layer-1 triggers
for p, i in enumerate(A_UNITS):
    w['W1'][i, 40 + p] = K_A
    w['W1'][i, 45] = K_A
    w['W1'][i, UP] = K_A * C_P
    w['b1'][i] = -K_A * TH_A
for p, i in enumerate(C_UNITS):
    w['W1'][i, 40 + p] = A_C
    w['W1'][i, UP] = B_C
    w['b1'][i] = -A_C - B_C * P0
# b2 compensation: repurposed L1 units sit at -1 when off
for i in L1U:
    w['b2'] += w0['W2'][:, i] * (m1[i] + 1.0)
# layer-2 gates
h1n = np.tanh(Ym @ w['W1'].T + w['b1'])
targets = {}
for p in range(4):
    v = np.zeros(4); v[(p + 1) % 4] = M / 2; targets[GA[p]] = (A_UNITS[p], v)
    v = np.zeros(4); v[(p + 2) % 4] = -M / 2; v[(p + 3) % 4] = -M / 2; targets[GC[p]] = (C_UNITS[p], v)
for j, (i, v) in targets.items():
    w['W2'][j, i] = G
    rest = h1n @ np.delete(w['W2'][j], i) if False else (h1n @ w['W2'][j] - G * h1n[:, i])
    w['b2'][j] = -np.median(rest)
    w['W3'][:, j] = v
    w['b3'] += w0['W3'][:, j] * m2[j] + v   # old mean contribution minus new off contribution (-v)

if __name__ == '__main__':
    out = os.path.join(HERE, 'weights.npz')
    tools.save(w, out)
    print('L1 triggers A', A_UNITS, 'C', C_UNITS, '| L2 gates A', GA, 'C', GC)
    print(tools.edit_size(tools.load(out), w0))
