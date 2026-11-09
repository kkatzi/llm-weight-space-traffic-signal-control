"""Patch = (1) W2 imputation from E2 + (2) hold ramps folded into the t_green column.
Seen-by-network inputs (x11, x23 are dead, so their W1 columns are reused through other inputs):
   q_W2_hat = q_E2 + gW * (t_green - 0.2),   d_W2_hat = d_E2 + gW * (t_green - 0.2)
   q/d of N0, N1, S0, S1 seen as  x + gN * (t_green - 0.2)
Honoured decisions have t_green >= 0.2 (12 s), so the ramps are 0 at the usual first decision and only act
while a green is held (in practice only phase 2 is ever held).
Parameter cost: W1[:,5], W1[:,17] (128) + W1[:,45], b1 (128) = 256."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
NS_THRU = [0, 1, 6, 7, 12, 13, 18, 19]

def build(gW=0.0, gN=0.0, t0=0.2, w0=None):
    w0 = tools.load('start_weights.npz') if w0 is None else w0
    w = {k: v.copy() for k, v in w0.items()}
    W1 = w0['W1']
    w['W1'][:, 5] += W1[:, 11]
    w['W1'][:, 17] += W1[:, 23]
    r = gW * (W1[:, 11] + W1[:, 23]) + gN * W1[:, NS_THRU].sum(1)
    if np.any(r != 0):
        w['W1'][:, 45] += r
        w['b1'] -= t0 * r
    return w

def imputed(X, gW=0.0, gN=0.0, t0=0.2):
    """What the ORIGINAL network sees when the patched network reads failed observations X."""
    X = np.array(X, dtype=np.float64, copy=True)
    ramp = X[:, 45] - t0
    X[:, 11] = X[:, 5] + gW * ramp; X[:, 23] = X[:, 17] + gW * ramp
    for i in NS_THRU: X[:, i] += gN * ramp
    return X

if __name__ == '__main__':
    gW, gN, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    w = build(gW, gN)
    tools.save(w, out)
    print(out, tools.edit_size(tools.load(out), tools.load('start_weights.npz')))
