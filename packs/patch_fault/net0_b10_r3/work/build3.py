"""Variant builder v3.
proxy:     x11 := x5 (q_E2), x23 := x17 (d_E2)                    -> W1[:,5] += W1[:,11]; W1[:,17] += W1[:,23]
phantom:   imputed inputs get + c*(t_green - tg0) on chosen lanes  -> W1[:,45] += c*sum(W1[:,lanes]); b1 -= c*tg0*sum(...)
           c3 for the dead W2 lane (q and d), cN for the N/S through lanes (q and d of N0,N1,S0,S1)
"""
import sys; sys.path.insert(0, '.')
import tools, numpy as np
NS = [0, 1, 6, 7, 12, 13, 18, 19]          # q/d of N0, N1, S0, S1

def build(c3=0.0, cN=0.0, tg0=0.2, ns_lanes=NS, b3_delta=None):
    w0 = tools.load("start_weights.npz")
    w = {k: v.copy() for k, v in w0.items()}
    W = w0['W1']
    w['W1'][:, 5] += W[:, 11]
    w['W1'][:, 17] += W[:, 23]
    v = c3 * (W[:, 11] + W[:, 23]) + cN * W[:, ns_lanes].sum(1)
    if np.any(v != 0):
        w['W1'][:, 45] += v
        w['b1'] -= tg0 * v
    if b3_delta is not None:
        w['b3'] += np.asarray(b3_delta, float)
    return w, w0

if __name__ == "__main__":
    import json
    cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    w, w0 = build(**cfg)
    tools.save(w, sys.argv[1])
    print(cfg, tools.edit_size(tools.load(sys.argv[1]), w0))
