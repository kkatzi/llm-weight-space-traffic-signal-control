"""Variant: shrinkage imputation  W2_hat = rho*E2 + (1-rho)*mean(E2)  (q and d), plus the 60-s stall breaker
(unit 57) re-derived around the new idle baseline. rho from corr(N2,S2) on recorded states (analogue pair)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
NS_THRU = [0, 1, 6, 7, 12, 13, 18, 19]; DEAD = [11, 23]

def build(rho_q, rho_d, m_q, m_d, vW=0.67, vN=0.35, J=57, K=200.0, thr=1.975):
    w0 = tools.load('start_weights.npz'); W1 = w0['W1']
    S = tools.states(); X = S['obs'].astype(np.float64)
    w = {k: v.copy() for k, v in w0.items()}
    w['W1'][:, 5] += rho_q * W1[:, 11]
    w['W1'][:, 17] += rho_d * W1[:, 23]
    base_q, base_d = (1 - rho_q) * m_q, (1 - rho_d) * m_d
    w['b1'] = w0['b1'] + base_q * W1[:, 11] + base_d * W1[:, 23]
    h1, _ = tools.hidden(w, X[S['controller'] == 0]); mu = h1[:, J].mean()
    xe = np.zeros(46); xe[42] = 1; xe[44] = 1; xe[45] = 1.0; xe[11] = base_q; xe[23] = base_d
    xi = xe.copy(); xi[11] = max(vW, base_q); xi[23] = max(vW, base_d); xi[NS_THRU] += vN
    dh1 = np.tanh(W1 @ xi + w0['b1']) - np.tanh(W1 @ xe + w0['b1']); dh1[J] = 0
    shift = w0['W2'] @ dh1; on = np.tanh(K * (2 - thr))
    live = [i for i in range(46) if i not in DEAD]
    w['W1'][J, live] = 0.0; w['W1'][J, 45] = K; w['W1'][J, 42] = K; w['b1'][J] = -K * thr
    newcol = shift / (2 * on)
    w['b2'] = w0['b2'] + w0['W2'][:, J] * mu + newcol * on
    w['W2'][:, J] = newcol
    return w

if __name__ == '__main__':
    rq, rd, mq, md, out = map(float, sys.argv[1:5]) + [sys.argv[5]] if False else (*map(float, sys.argv[1:5]), sys.argv[5])
    w = build(rq, rd, mq, md); tools.save(w, out)
    print(out, tools.edit_size(tools.load(out), tools.load('start_weights.npz')))
