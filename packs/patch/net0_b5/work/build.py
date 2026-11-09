"""Build the edited network by construction (no fitting).

Mechanism: 'empty-phase penalty'.
  For each phase p, a repurposed layer-1 unit J[p] detects that phase p has NO vehicles on any of its stop-line or
  upstream lanes (density sum == 0). A repurposed layer-2 unit K[p] turns that into a clean +-1 flag, and W3 maps the
  flag onto logit p: -M/2 when empty, +M/2 otherwise (uniform across phases when nothing is empty, so no effect).
  Result: the controller never gives green to (or stays on) a phase with no vehicles while another phase has some.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np

SL = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}      # stop-line lanes per phase
UP = {0: [0, 1, 4, 5], 1: [1, 5], 2: [2, 3, 6, 7], 3: [3, 7]}       # upstream lanes per phase
Q0, D0, UQ0, UD0 = 0, 12, 24, 32

def build(J=(60, 57, 32, 13), K=(37, 0, 11, 5), Kd=300.0, beta=-1.0, half_swing=10.0, M=24.0,
          comp_b2=True, comp_b3=True, w0=None, X=None):
    w0 = tools.load('start_weights.npz') if w0 is None else w0
    X = tools.states()['obs'].astype(np.float64) if X is None else X
    w = {k: v.copy() for k, v in w0.items()}
    h1o, h2o = tools.hidden(w0, X)
    mu1 = h1o.mean(0)
    # ---- layer 1: emptiness detectors
    for p in range(4):
        j = J[p]
        keep = np.zeros(46, bool)
        for l in SL[p]: keep[Q0 + l] = True        # halting counts: zero whenever density is zero -> keep old
        for u in UP[p]: keep[UQ0 + u] = True
        row = np.where(keep, w0['W1'][j], 0.0)
        for l in SL[p]: row[D0 + l] = -Kd
        for u in UP[p]: row[UD0 + u] = -Kd
        w['W1'][j] = row
        w['b1'][j] = beta
    # ---- layer 2 bias compensation for the replaced layer-1 units (normal value is -1)
    if comp_b2:
        for j in J:
            w['b2'] += w0['W2'][:, j] * (mu1[j] - (-1.0))
    # ---- layer 2: flag units
    delta = np.tanh(beta) + 1.0
    c = 2 * half_swing / delta
    h1n = np.tanh(X @ w['W1'].T + w['b1'])
    for p in range(4):
        k, j = K[p], J[p]
        w['W2'][k, j] = c
        rest = h1n @ w['W2'][k].T - c * h1n[:, j]
        w['b2'][k] = -rest.mean() + c * (1 - delta / 2)
    # ---- output: replace lost layer-2 units by their mean, add penalty
    mu2 = h2o.mean(0)
    for p in range(4):
        k = K[p]
        if comp_b3:
            w['b3'] += w0['W3'][:, k] * mu2[k]
        col = np.zeros(4); col[p] = -M / 2
        w['W3'][:, k] = col
    w['b3'] -= M / 2          # cosmetic: cancel the uniform +M/2 of the normal state
    return w

def demand(X):
    d = X[:, D0:D0 + 12]; ud = X[:, UD0:UD0 + 8]
    return np.stack([d[:, SL[p]].sum(1) + ud[:, UP[p]].sum(1) for p in range(4)], 1)

if __name__ == '__main__':
    w0 = tools.load('start_weights.npz')
    w = build()
    print(tools.edit_size(w, w0))
