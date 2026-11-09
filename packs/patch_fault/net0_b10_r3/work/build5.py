"""General builder (final family). All edits fold an imputation of the two dead inputs (and a time phantom on the
N/S through lanes) into layer 1; no other layer is touched.

    q_W2_hat = alpha*q_E2 + (1-alpha)*mq + k*q_W1u + c3*(t_green - 0.2)
    d_W2_hat = alpha*d_E2 + (1-alpha)*md + k*q_W1u + c3*(t_green - 0.2)
    q/d of N0, N1, S0, S1  +=  cN*(t_green - 0.2)

Folding: an input x_i replaced by sum_j a_ij x_j + b_i  ->  W1[:,j] += a_ij*W1orig[:,i],  b1 += b_i*W1orig[:,i].
Columns touched: W1[:,5], W1[:,17], W1[:,31], W1[:,45] and b1  -> 5*64 = 320 parameters.
"""
import sys; sys.path.insert(0, '.')
import tools, numpy as np
NS = [0, 1, 6, 7, 12, 13, 18, 19]

def build(alpha=1.0, mq=0.32, md=0.40, k=6.0, c3=1.0, cN=0.5, tg0=0.2):
    w0 = tools.load("start_weights.npz")
    w = {kk: v.copy() for kk, v in w0.items()}
    W = w0['W1']
    uq, ud = W[:, 11], W[:, 23]
    w['W1'][:, 5] += alpha * uq
    w['W1'][:, 17] += alpha * ud
    w['b1'] += (1 - alpha) * (mq * uq + md * ud)
    w['W1'][:, 31] += k * (uq + ud)
    v = c3 * (uq + ud) + cN * W[:, NS].sum(1)
    w['W1'][:, 45] += v
    w['b1'] -= tg0 * v
    return w, w0

if __name__ == "__main__":
    import json
    cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    w, w0 = build(**cfg)
    tools.save(w, sys.argv[1])
    print(cfg, tools.edit_size(tools.load(sys.argv[1]), w0))
