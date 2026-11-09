"""Max-green safety gate with NON-destructive second-layer carriers.

Gate unit (first layer, one per gated phase p; repurposed unit T[p]):
    h1[T] = tanh(old_preact + A*phase_p + B*t_green - C)
    == -1 in all recorded states (greens never exceed 37 s), +1 once phase p has been green for ~>47 s.
Its old role is replaced by its mean: b2[i] += W2_old[i,T]*mean_old(h1[T]) + W2_new[i,T]   (normal h1[T] = -1)
Carriers: existing second-layer units k get W2[k,T] = GW*s_k. In normal states h1[T] = -1 exactly, and the b2 term
above cancels -GW*s_k exactly, so the carriers behave as before. When the gate fires, carrier k is driven to s_k,
changing the logits by W3[:,k]*(s_k - h2_k). Carriers and signs are chosen by a closed-form score: how much
driving unit k to s_k lowers logit p relative to the mean of the other logits in the gated situation (empty junction,
phase p green for 60 s). No output-layer weights change.
"""
import numpy as np, sys, copy, json
sys.path.insert(0, '.')
import tools

W0 = tools.load('start_weights.npz')
A, B, C, GW = 21.0, 60.0, 66.0, 10.0

def gated_state(p):
    x = np.zeros(46); x[40 + p] = 1; x[44] = 1; x[45] = 1.0
    return x

def choose_carriers(p, n):
    _, h2 = tools.hidden(W0, gated_state(p)[None])
    h2 = h2[0]
    W3 = W0['W3']
    others = [q for q in range(4) if q != p]
    rel = W3[others].mean(0) - W3[p]             # >0: raising h2_k raises others relative to p
    s = np.sign(rel)
    score = np.abs(rel) * np.abs(s - h2)          # achievable suppression of p
    order = np.argsort(-score)[:n]
    return [(int(k), float(s[k]), float(score[k])) for k in order]

def build(phases=(2, 3), t_units=(31, 0), n_carriers=12):
    S = tools.states(); X = S['obs'].astype(np.float64)
    h1o, _ = tools.hidden(W0, X)
    w = copy.deepcopy(W0)
    used = set()
    plan = {}
    for p, j in zip(phases, t_units):
        w['W1'][j, 40 + p] = W0['W1'][j, 40 + p] + A
        w['W1'][j, 45] = W0['W1'][j, 45] + B
        w['b1'][j] = W0['b1'][j] - C
        car = choose_carriers(p, n_carriers)
        plan[p] = car
        for k, s, _ in car:
            w['W2'][k, j] = GW * s
    for i in range(64):
        w['b2'][i] = W0['b2'][i] + sum(W0['W2'][i, j] * h1o[:, j].mean() + w['W2'][i, j] for j in t_units)
    return w, plan

if __name__ == '__main__':
    name = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    w, plan = build(n_carriers=n)
    for p, car in plan.items(): print('phase', p, 'carriers (unit, sign, score):', [(k, int(s), round(sc, 2)) for k, s, sc in car])
    out = f'work/{name}.npz'; tools.save(w, out); wb = tools.load(out)
    print(out, tools.edit_size(wb, W0))
    S = tools.states(); X = S['obs'].astype(np.float64); c = S['controller']; ok = X[:, 44] > 0.5
    g0 = tools.greedy(W0, X); g1 = tools.greedy(wb, X)
    print('  ok-decision change rate: all', round(float((g0[ok] != g1[ok]).mean()), 4), ' start-net', round(float((g0[ok & (c==0)] != g1[ok & (c==0)]).mean()), 4))
    for p in (2, 3):
        x = gated_state(p)
        print(f'  gated empty state, phase {p}: original logits', np.round(tools.forward(W0, x)[0], 2), ' new', np.round(tools.forward(wb, x)[0], 2))
