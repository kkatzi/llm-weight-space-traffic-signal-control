"""Time-gated extension bonus. Same decoder construction as build.py, but the phase-level unit K also reads t_green:
   a_K = L[phase] - GAMMA*(t_green - 0.2)
With GAMMA=36 the 4 levels are exact at t_green = 12 s (the normal first honoured decision) and at >= 17 s every
phase collapses to the bottom level, which carries no bonus. Phase order on the levels: ORDER[0] is bottom (no bonus).
=> logits += beta * e_{current phase}  only when t_green == 12 s and current phase != ORDER[0]; else nothing.
"""
import sys, numpy as np, tools
K = 31; J = [25, 35, 54]; G = 15.0; GAMMA = 36.0
ORDER = [2, 0, 1, 3]                      # phase at level 0 (bottom, no bonus), 1, 2, 3
LEVELS = np.array([-1.5, -0.5, 0.5, 1.5])
THETA = [(np.tanh(LEVELS[i-1]) + np.tanh(LEVELS[i])) / 2 for i in (1, 2, 3)]

def build(beta, w0=None, order=ORDER, gamma=GAMMA):
    w0 = tools.load("start_weights.npz") if w0 is None else w0
    w = {k: v.copy() for k, v in w0.items()}
    w["W1"][K, :] = 0.0
    for lvl, p in enumerate(order):
        w["W1"][K, 40 + p] = LEVELS[lvl]
    w["W1"][K, 45] = -gamma; w["b1"][K] = gamma * 0.2
    w["W2"][:, K] = 0.0
    E = np.eye(4)
    f = [np.zeros(4)] + [beta * E[p] for p in order[1:]]      # bonus per level; level 0 none
    for i, j in enumerate(J, start=1):
        w["W2"][j, :] = 0.0; w["W2"][j, K] = G; w["b2"][j] = -G * THETA[i-1]
        w["W3"][:, j] = (f[i] - f[i-1]) / 2
    w["b3"] = w["b3"] + f[0] + sum((f[i] - f[i-1]) / 2 for i in (1, 2, 3))
    return w

if __name__ == "__main__":
    beta = float(sys.argv[1]); out = sys.argv[2]
    w = build(beta); tools.save(w, out)
    print(out, tools.edit_size(w, tools.load("start_weights.npz")))
