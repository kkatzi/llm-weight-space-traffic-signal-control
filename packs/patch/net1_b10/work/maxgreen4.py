"""Variant of maxgreen3.py: carriers scored for suppressing the gated phase p while raising the other three
phases EVENLY (so that, once p is suppressed, the network's own halting-based preferences pick the next phase).
score_k = [mean_{q!=p} e_k[q] - e_k[p]] - LAMBDA * [max_{q!=p} e_k[q] - min_{q!=p} e_k[q]],  e_k = W3[:,k]*(s_k - h2_k)
evaluated in the gated situation (empty junction, phase p green 60 s)."""
import numpy as np, sys, copy
sys.path.insert(0, '.')
import tools
import importlib.util
spec = importlib.util.spec_from_file_location('mg3', 'work/maxgreen3.py'); mg3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(mg3)
W0 = mg3.W0
LAMBDA = 1.0

def choose_carriers(p, n):
    _, h2 = tools.hidden(W0, mg3.gated_state(p)[None]); h2 = h2[0]
    W3 = W0['W3']; others = [q for q in range(4) if q != p]
    best = []
    for k in range(64):
        for s in (1.0, -1.0):
            e = W3[:, k] * (s - h2[k])
            sc = e[others].mean() - e[p] - LAMBDA * (e[others].max() - e[others].min())
            best.append((sc, k, s))
    best.sort(reverse=True)
    out, seen = [], set()
    for sc, k, s in best:
        if k in seen: continue
        seen.add(k); out.append((int(k), float(s), float(sc)))
        if len(out) == n: break
    return out

mg3.choose_carriers = choose_carriers

if __name__ == '__main__':
    name = sys.argv[1]; n = int(sys.argv[2]); LAMBDA = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    w, plan = mg3.build(n_carriers=n)
    for p, car in plan.items(): print('phase', p, 'carriers:', [(k, int(s), round(sc, 2)) for k, s, sc in car])
    out = f'work/{name}.npz'; tools.save(w, out); wb = tools.load(out)
    print(out, tools.edit_size(wb, W0))
    S = tools.states(); X = S['obs'].astype(np.float64); c = S['controller']; ok = X[:, 44] > 0.5
    g0 = tools.greedy(W0, X); g1 = tools.greedy(wb, X)
    print('  ok-decision change rate: all', round(float((g0[ok] != g1[ok]).mean()), 4), ' start-net', round(float((g0[ok & (c==0)] != g1[ok & (c==0)]).mean()), 4))
    for p in (2, 3):
        x = mg3.gated_state(p)
        print(f'  gated empty state, phase {p}: original logits', np.round(tools.forward(W0, x)[0], 2), ' new', np.round(tools.forward(wb, x)[0], 2))
