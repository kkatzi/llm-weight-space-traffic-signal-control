import numpy as np, sys, copy
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); c = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
g = tools.greedy(w, X)
LEFT_Q = [2, 5, 8, 11]; LEFT_D = [14, 17, 20, 23]
m = ok & (c == 0)
def scaled(alpha, cols, topk=None):
    w2 = copy.deepcopy(w)
    for col in cols:
        colv = w['W1'][:, col]
        if topk is None: idx = np.arange(64)
        else: idx = np.argsort(-np.abs(colv))[:topk]
        w2['W1'][idx, col] = colv[idx] * alpha
    return w2
for alpha in [0.6, 0.8, 1.2, 1.4]:
    full = scaled(alpha, LEFT_Q + LEFT_D)
    gf = tools.greedy(full, X)
    ch = m & (gf != g)
    print(f'alpha {alpha}: full scaling (512 params) changes {ch.mean()/m.mean():.3f} of ok decisions; new holds {(ch & (gf==ph)).sum()} lost holds {(ch & (g==ph)).sum()}')
    # breakdown: through->left / left->through
    thr = lambda a: (a % 2 == 0)
    print('     pick through->left', (ch & thr(g) & ~thr(gf)).sum(), ' left->through', (ch & ~thr(g) & thr(gf)).sum(), ' same type', (ch & (thr(g) == thr(gf))).sum())
    for k in [16, 24, 32, 40, 46]:
        part = scaled(alpha, LEFT_Q + LEFT_D, topk=k)
        gp = tools.greedy(part, X)
        print(f'     top-{k} per column ({8*k} params): changes {(m & (gp != g)).mean()/m.mean():.3f}, agrees with full scaling {(gp[m]==gf[m]).mean():.3f}')
    qo = scaled(alpha, LEFT_Q); go = tools.greedy(qo, X)
    print(f'     q-cols only (256 params): changes {(m & (go != g)).mean()/m.mean():.3f}, agrees with full {(go[m]==gf[m]).mean():.3f}')
