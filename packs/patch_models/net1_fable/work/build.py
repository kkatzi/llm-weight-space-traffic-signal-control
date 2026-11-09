"""Construct the edited network: a constant 'stay bonus' beta added to the logit of the current green phase.

Mechanism (all values set by hand, no fitting):
  layer-1 unit K  : h1_K = tanh(-1.5*phase0 - 0.5*phase1 + 0.5*phase2 + 1.5*phase3)  -> 4 levels -0.905,-0.462,0.462,0.905
  its old W2 column is zeroed (severed), so the rest of the net no longer sees it
  layer-2 units J1,J2,J3: h2_Ji = tanh(G*(h1_K - theta_i)), theta = (-0.683, 0, 0.683) -> step functions [phase>=i]
  W3[:,Ji] = beta/2 * (e_i - e_{i-1}),  b3 += beta/2 * (e_0 + e_3)
  => logits += beta * e_{current phase}
"""
import sys, numpy as np, tools
K = 31; J = [25, 35, 54]; G = 15.0
LEVELS = np.array([-1.5, -0.5, 0.5, 1.5])
THETA = [(np.tanh(LEVELS[i-1]) + np.tanh(LEVELS[i])) / 2 for i in (1, 2, 3)]

def build(beta, w0=None):
    w0 = tools.load("start_weights.npz") if w0 is None else w0
    w = {k: v.copy() for k, v in w0.items()}
    w["W1"][K, :] = 0.0; w["W1"][K, 40:44] = LEVELS; w["b1"][K] = 0.0
    w["W2"][:, K] = 0.0
    E = np.eye(4)
    for i, j in enumerate(J, start=1):
        w["W2"][j, :] = 0.0; w["W2"][j, K] = G; w["b2"][j] = -G * THETA[i-1]
        w["W3"][:, j] = beta / 2 * (E[i] - E[i-1])
    w["b3"] = w["b3"] + beta / 2 * (E[0] + E[3])
    return w

if __name__ == "__main__":
    beta = float(sys.argv[1]); out = sys.argv[2]
    w = build(beta); tools.save(w, out)
    print(out, tools.edit_size(w, tools.load("start_weights.npz")))
