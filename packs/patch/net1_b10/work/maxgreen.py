"""Max-green safety gate, built by construction.

For each phase p we repurpose one first-layer unit T[p] and one second-layer unit G[p] (chosen among the units whose
replacement by their mean activation changes the fewest recorded decisions).
  h1[T[p]] = tanh(old_preact + A*phase_p + B*t_green - C)   -> -1 in normal operation, +1 once phase p has been green
                                                               for >= ~45-50 s (normal greens never exceed 37 s)
  h2[G[p]] = tanh(rest + GW*h1[T[p]] - GB)                   -> -1 normally, +1 when the gate fires
  logit[p] += 2*SUPP*... : W3[p, G[p]] = -SUPP  (logit p drops by 2*SUPP when the gate fires)
Compensation so normal behaviour is unchanged (up to mean-ablation of the 8 repurposed units):
  b2[i] += sum_p W2[i, T[p]] * (mean_old(h1[T[p]]) + 1)        for all non-gate units i
  b3    += sum_p W3_old[:, G[p]]*mean_old(h2[G[p]]) + W3_new[:, G[p]]   (normal h2[G[p]] = -1)
"""
import numpy as np, sys, copy
sys.path.insert(0, '.')
import tools

W0 = tools.load('start_weights.npz')
T_UNITS = [31, 0, 59, 4]      # first-layer units repurposed as time gates, phase 0..3
G_UNITS = [25, 15, 1, 53]     # second-layer units repurposed as gate carriers, phase 0..3
A, B, C = 21.0, 60.0, 66.0    # gate fires when phase p and t_green >~ 0.78 (47 s); off for t_green <= 0.62 (37 s)
GW = 15.0                     # second-layer gate weight (b2 of the gate unit set so normal pre-activation << 0)
SUPP = 10.0                   # W3[p, G[p]] = -SUPP: logit p drops by 2*SUPP when the gate fires

def build(base=None):
    base = W0 if base is None else base
    S = tools.states(); X = S['obs'].astype(np.float64)
    h1o, h2o = tools.hidden(W0, X)
    w = copy.deepcopy(base)
    # first layer gates
    for p, j in enumerate(T_UNITS):
        w['W1'][j, 40 + p] = W0['W1'][j, 40 + p] + A
        w['W1'][j, 45] = W0['W1'][j, 45] + B
        w['b1'][j] = W0['b1'][j] - C
    # b2 compensation for non-gate units (replace old h1[T] by its mean; new normal value is -1)
    comp = np.zeros(64)
    for j in T_UNITS:
        comp += W0['W2'][:, j] * (h1o[:, j].mean() + 1.0)
    for i in range(64):
        if i not in G_UNITS:
            w['b2'][i] = base['b2'][i] + comp[i]
    # second layer gate carriers
    for p, (j, k) in enumerate(zip(T_UNITS, G_UNITS)):
        w['W2'][k, j] = GW
        w['b2'][k] = W0['b2'][k]          # keep; the -GW from h1=-1 already pushes it negative
    # output layer
    b3 = base['b3'].copy()
    for p, k in enumerate(G_UNITS):
        w['W3'][p, k] = -SUPP
        b3 += W0['W3'][:, k] * h2o[:, k].mean() + w['W3'][:, k]     # normal h2[k] = -1 -> contributes -W3new
    w['b3'] = b3
    return w

if __name__ == '__main__':
    w = build()
    tools.save(w, 'work/v3_maxgreen.npz')
    wb = tools.load('work/v3_maxgreen.npz')
    print(tools.edit_size(wb, W0))
    S = tools.states(); X = S['obs'].astype(np.float64); c = S['controller']
    ok = X[:, 44] > 0.5
    g0 = tools.greedy(W0, X); g1 = tools.greedy(wb, X)
    for ci in range(3):
        m = ok & (c == ci)
        print(S['controller_names'][ci], 'ok-decision change rate', round(float((g0[m] != g1[m]).mean()), 4), ' all-decision', round(float((g0[c==ci] != g1[c==ci]).mean()), 4))
    h1n, h2n = tools.hidden(wb, X)
    print('gate h1 max over recorded states', h1n[:, T_UNITS].max(0), ' gate h2 max', h2n[:, G_UNITS].max(0))
    a1 = X @ wb['W1'].T + wb['b1']; a2 = h1n @ wb['W2'].T + wb['b2']
    print('gate h1 pre-act max', a1[:, T_UNITS].max(0), ' gate h2 pre-act max', a2[:, G_UNITS].max(0))
