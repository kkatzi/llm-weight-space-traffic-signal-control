"""Final patch v2 = make_final.build (W2<-E2 imputation + 60-s stall breaker on unit 57)
 + spillback term: q_W2_hat, d_W2_hat += beta * q_W1u   (W1[:,31] += beta*(W1[:,11]+W1[:,23]), rows != 57; 63 params)
beta from recorded statistics: with an upstream queue on the SAME approach the pocket is ~0.8 full, while the E2 proxy
reads ~0.55 when W1u is queued; gap ~0.25 at a typical q_W1u of 0.1-0.15 -> beta ~1.5-2.5 (1.5 = conservative)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
from make_final import build as build1

def build(vW=0.67, vN=0.35, beta=1.5, J=57):
    w0 = tools.load('start_weights.npz')
    w = build1(vW, vN, J=J, w0=w0)
    rows = [j for j in range(64) if j != J]
    w['W1'][rows, 31] += beta * (w0['W1'][rows, 11] + w0['W1'][rows, 23])
    return w

if __name__ == '__main__':
    beta, out = float(sys.argv[1]), sys.argv[2]
    w = build(beta=beta)
    tools.save(w, out)
    print(out, tools.edit_size(tools.load(out), tools.load('start_weights.npz')))
    S = tools.states(); X = S['obs'].astype(np.float64); ok = X[:, 44] > 0.5
    g1 = tools.greedy(tools.load('work/final_w067_n035.npz'), X); g2 = tools.greedy(w, X)
    for c in range(3):
        m = ok & (S['controller'] == c)
        print(' ctrl', c, 'changed', int((g1[m] != g2[m]).sum()), '/', m.sum(), 'phase3 share', round((g1[m] == 3).mean(), 3), '->', round((g2[m] == 3).mean(), 3))
