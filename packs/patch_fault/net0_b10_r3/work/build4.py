"""Variant builder v4 (all edits are input imputations folded into layer 1).
Imputed dead inputs:
    q_W2_hat = q_E2 + k*q_W1u + c3*(t_green - 0.2)
    d_W2_hat = d_E2 + k*q_W1u + c3*(t_green - 0.2)
Time phantom also on N/S through lanes (q,d of N0,N1,S0,S1): + cN*(t_green - 0.2)
Folding x_i_hat = sum_j a_ij x_j + b_i into layer 1: W1[:,j] += a_ij * W1[:,i] (orig), b1 += b_i * W1[:,i].
"""
import sys; sys.path.insert(0, '.')
import tools, numpy as np
NS = [0, 1, 6, 7, 12, 13, 18, 19]

def build(c3=1.0, cN=0.5, k=0.0, tg0=0.2, k_col=31, delta=0.0):
    w0 = tools.load("start_weights.npz")
    w = {kk: v.copy() for kk, v in w0.items()}
    W = w0['W1']
    u = W[:, 11] + W[:, 23]                       # both dead inputs get the same additive terms
    w['W1'][:, 5] += W[:, 11]                     # q_W2 <- q_E2
    w['W1'][:, 17] += W[:, 23]                    # d_W2 <- d_E2
    v = c3 * u + cN * W[:, NS].sum(1)             # time phantom
    if np.any(v != 0):
        w['W1'][:, 45] += v
        w['b1'] -= tg0 * v
    if k:
        w['W1'][:, k_col] += k * u                # upstream spill-back evidence
    if delta:
        w['b1'] += delta * u
    return w, w0

if __name__ == "__main__":
    import json
    cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    w, w0 = build(**cfg)
    tools.save(w, sys.argv[1])
    print(cfg, tools.edit_size(tools.load(sys.argv[1]), w0))
