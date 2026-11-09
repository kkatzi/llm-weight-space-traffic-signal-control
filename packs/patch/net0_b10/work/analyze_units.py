import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
X = S['obs'].astype(np.float64)
a0 = tools.greedy(w, X); ok = X[:,44]>0.5
h1, h2 = tools.hidden(w, X)
def run_h1_const(k, c):
    H1 = h1.copy(); H1[:,k] = c
    H2 = np.tanh(H1 @ w['W2'].T + w['b2'])
    return (H2 @ w['W3'].T + w['b3']).argmax(1)
def run_h2_const(j, c, comp=True):
    H2 = h2.copy(); H2[:,j] = c
    b3 = w['b3'] + (w['W3'][:,j]*(h2[:,j].mean()-c) if comp else 0)
    return (H2 @ w['W3'].T + b3).argmax(1)
rows=[]
for k in range(64):
    best=None
    for c in (-1.0, 1.0):
        a = run_h1_const(k,c); d = (a!=a0)[ok].mean()
        if best is None or d<best[1]: best=(c,d)
    rows.append((best[1], k, best[0], h1[:,k].mean(), h1[:,k].std()))
rows.sort()
print('h1 units: frac ok-decisions changed when unit set to const c')
for r in rows[:12]: print('  k=%2d c=%+.0f changed=%.4f mean=%.2f std=%.2f'%(r[1],r[2],r[0],r[3],r[4]))
rows=[]
for j in range(64):
    best=None
    for c in (-1.0, 1.0):
        a = run_h2_const(j,c); d=(a!=a0)[ok].mean()
        if best is None or d<best[1]: best=(c,d)
    rows.append((best[1], j, best[0], h2[:,j].mean(), h2[:,j].std(), w['W3'][:,j]))
rows.sort()
print('h2 units: frac ok-decisions changed when unit set to const c (b3 compensated for mean)')
for r in rows[:12]: print('  j=%2d c=%+.0f changed=%.4f mean=%.2f std=%.2f W3col=%s'%(r[1],r[2],r[0],r[3],r[4],r[5]))
