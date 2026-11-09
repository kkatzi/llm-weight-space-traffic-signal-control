"""Build edited weights by construction.
Variant 'proxy': substitute the dead W2 inputs by the mirror lane E2 (same phase, same demand by symmetry):
  x11 (q_W2) := x5 (q_E2), x23 (d_W2) := x17 (d_E2)  ->  W1[:,5] += W1[:,11], W1[:,17] += W1[:,23]
Optional shrinkage alpha: x11 := alpha*x5 + (1-alpha)*mu_q, x23 := alpha*x17 + (1-alpha)*mu_d (mu into b1).
Optional b3 offsets.
"""
import sys; sys.path.insert(0, '.')
import tools, numpy as np

def build(alpha=1.0, mu_q=0.0, mu_d=0.0, b3_delta=None):
    w0 = tools.load("start_weights.npz")
    w = {k: v.copy() for k, v in w0.items()}
    w['W1'][:, 5] += alpha * w0['W1'][:, 11]
    w['W1'][:, 17] += alpha * w0['W1'][:, 23]
    if alpha < 1.0:
        w['b1'] += (1 - alpha) * (mu_q * w0['W1'][:, 11] + mu_d * w0['W1'][:, 23])
    if b3_delta is not None:
        w['b3'] += np.asarray(b3_delta, float)
    return w, w0

if __name__ == "__main__":
    import json
    cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    w, w0 = build(**cfg)
    print(tools.edit_size(w, w0))
    tools.save(w, sys.argv[1])
    # check after float32 round trip
    print(tools.edit_size(tools.load(sys.argv[1]), w0))
