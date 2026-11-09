"""Max-green safety gate (generalised): gate only the phases listed in PHASES.
Same construction as maxgreen.py (see its docstring)."""
import numpy as np, sys, copy, json
sys.path.insert(0, '.')
import tools

W0 = tools.load('start_weights.npz')

def build(phases, t_units, g_units, A=21.0, B=60.0, C=66.0, GW=15.0, SUPP=10.0):
    S = tools.states(); X = S['obs'].astype(np.float64)
    h1o, h2o = tools.hidden(W0, X)
    w = copy.deepcopy(W0)
    for p, j in zip(phases, t_units):
        w['W1'][j, 40 + p] = W0['W1'][j, 40 + p] + A
        w['W1'][j, 45] = W0['W1'][j, 45] + B
        w['b1'][j] = W0['b1'][j] - C
    comp = np.zeros(64)
    for j in t_units:
        comp += W0['W2'][:, j] * (h1o[:, j].mean() + 1.0)
    for i in range(64):
        if i not in g_units:
            w['b2'][i] = W0['b2'][i] + comp[i]
    for p, j, k in zip(phases, t_units, g_units):
        w['W2'][k, j] = GW
    b3 = W0['b3'].copy()
    for p, k in zip(phases, g_units):
        w['W3'][p, k] = -SUPP
        b3 += W0['W3'][:, k] * h2o[:, k].mean() + w['W3'][:, k]
    w['b3'] = b3
    return w

if __name__ == '__main__':
    name = sys.argv[1]
    phases = json.loads(sys.argv[2]); tu = json.loads(sys.argv[3]); gu = json.loads(sys.argv[4])
    w = build(phases, tu, gu)
    out = f'work/{name}.npz'
    tools.save(w, out); wb = tools.load(out)
    print(out, tools.edit_size(wb, W0))
    S = tools.states(); X = S['obs'].astype(np.float64); c = S['controller']; ok = X[:, 44] > 0.5
    g0 = tools.greedy(W0, X); g1 = tools.greedy(wb, X)
    print('  ok-decision change rate: all', round(float((g0[ok] != g1[ok]).mean()), 4), ' start-net', round(float((g0[ok & (c==0)] != g1[ok & (c==0)]).mean()), 4))
