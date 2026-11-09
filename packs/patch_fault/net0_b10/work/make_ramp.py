"""Variant B: impute the dead W2 detector as  W2_hat = E2 + gamma * (t_green - 0.2)  (same for queue and density).
At an honoured decision t_green >= 0.2 (12 s), so the ramp is 0 at the normal first decision and grows only
while a green is held. Folding into layer 1 (x11 and x23 are always 0, so their columns are free to read):
  W1[:,5]  += W1[:,11]           (q_W2 <- q_E2)
  W1[:,17] += W1[:,23]           (d_W2 <- d_E2)
  W1[:,45] += gamma*(W1[:,11] + W1[:,23])
  b1       -= 0.2*gamma*(W1[:,11] + W1[:,23])
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools

def build(gamma, w0=None, t0=0.2):
    w0 = tools.load('start_weights.npz') if w0 is None else w0
    w = {k: v.copy() for k, v in w0.items()}
    c11, c23 = w0['W1'][:, 11], w0['W1'][:, 23]
    w['W1'][:, 5] += c11
    w['W1'][:, 17] += c23
    if gamma != 0:
        w['W1'][:, 45] += gamma * (c11 + c23)
        w['b1'] -= t0 * gamma * (c11 + c23)
    return w

if __name__ == '__main__':
    gamma = float(sys.argv[1]); out = sys.argv[2]
    w = build(gamma)
    tools.save(w, out)
    print(out, tools.edit_size(tools.load(out), tools.load('start_weights.npz')))
