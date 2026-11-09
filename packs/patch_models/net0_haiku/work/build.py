"""Build candidate readout edit: add alpha * (class-mean of h2 for states where phase k has the largest served
stop-line queue, minus overall mean of h2 on ok=1 states) to W3[k]. Only W3 changes (<=256 entries).
Statistics only (means of recorded hidden activations); no regression, no gradient, no search over weights."""
import sys
sys.path.insert(0, '/home/traffic-ntua/wsp_v3_agent_runs/patch_models/net0_haiku')
import numpy as np, tools

SER = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}   # stop-line queue lanes served by each phase


def build(alpha, out=None):
    w = tools.load('start_weights.npz')
    S = tools.states(); X = S['obs']
    ok = X[:, 44] > 0.5
    _, h2 = tools.hidden(w, X)
    Q = np.stack([X[:, SER[k]].sum(1) for k in range(4)], 1)
    top = Q.argmax(1)
    strong = ok & (Q.max(1) > 0) & (np.sort(Q, 1)[:, -1] > np.sort(Q, 1)[:, -2])   # unique max, nonzero
    mu_all = h2[ok].mean(0)
    W3 = w['W3'].copy()
    for k in range(4):
        m = strong & (top == k)
        W3[k] += alpha * (h2[m].mean(0) - mu_all)
    w2 = dict(w); w2['W3'] = W3
    if out:
        tools.save(w2, out)
    return w2


if __name__ == '__main__':
    base = tools.load('start_weights.npz')
    S = tools.states(); X = S['obs']; ok = X[:, 44] > 0.5
    g0 = tools.greedy(base, X)
    for a in [0.5, 1.0, 2.0]:
        w2 = build(a)
        g = tools.greedy(w2, X)
        print('alpha', a, 'changed decisions (ok=1):', round(float((g[ok] != g0[ok]).mean()), 3),
              'edit', tools.edit_size(w2, base))
    lg = tools.forward(base, X)
    print('logit std base', lg.std(0).round(2))


def build_stay(beta, out=None):
    """Stickiness candidate: W3[k] += beta * (mean h2 | current green is k, ok=1  -  mean h2 | ok=1).
    Statistics of recorded hidden activations only."""
    w = tools.load('start_weights.npz')
    S = tools.states(); X = S['obs']
    ok = X[:, 44] > 0.5
    _, h2 = tools.hidden(w, X)
    cur = X[:, 40:44].argmax(1)
    mu_all = h2[ok].mean(0)
    W3 = w['W3'].copy()
    for k in range(4):
        m = ok & (cur == k)
        W3[k] += beta * (h2[m].mean(0) - mu_all)
    w2 = dict(w); w2['W3'] = W3
    if out:
        tools.save(w2, out)
    return w2
