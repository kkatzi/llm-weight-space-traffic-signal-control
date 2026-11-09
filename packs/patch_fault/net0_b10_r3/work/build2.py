"""Variant builder.
proxy:   x11 := x5 (q_E2),  x23 := x17 (d_E2)                       W1[:,5] += W1[:,11]; W1[:,17] += W1[:,23]
phantom: both W2 inputs also get + c*(t_green - tg0)                W1[:,45] += c*(W1[:,11]+W1[:,23]); b1 -= c*tg0*(W1[:,11]+W1[:,23])
"""
import sys; sys.path.insert(0, '.')
import tools, numpy as np

def build(c=0.0, tg0=0.2, b3_delta=None):
    w0 = tools.load("start_weights.npz")
    w = {k: v.copy() for k, v in w0.items()}
    u11, u23 = w0['W1'][:, 11], w0['W1'][:, 23]
    w['W1'][:, 5] += u11
    w['W1'][:, 17] += u23
    if c:
        w['W1'][:, 45] += c * (u11 + u23)
        if tg0:
            w['b1'] -= c * tg0 * (u11 + u23)
    if b3_delta is not None:
        w['b3'] += np.asarray(b3_delta, float)
    return w, w0

if __name__ == "__main__":
    import json
    cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    w, w0 = build(**cfg)
    tools.save(w, sys.argv[1])
    print(cfg, tools.edit_size(tools.load(sys.argv[1]), w0))
