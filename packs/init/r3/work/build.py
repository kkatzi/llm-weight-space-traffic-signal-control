"""Build the starting network by hand from the rule in rule.py (no fitting).

V_p = rule score of phase p in vehicles (current-phase bonus included), a fixed linear function of x (rule.py).

Layer 1, units 0..11: unit 3p+j = tanh(a_j * V_p - c_j) for the three (a_j, c_j) in UNITS:
                        j=0 (0.30, 0.0): steep, resolves 0..6 vehicles
                        j=1 (0.30, 1.2): steep, centred at 4 vehicles
                        j=2 (0.08, 1.2): shallow "backbone", keeps V ordered up to ~40 vehicles
Layer 2, units 0..11: unit 3p+j = tanh(K2 * h1[3p+j])  (monotone copy; W2 from every other h1 unit is 0).
Output:               logit_p = BETA * sum_j h2[3p+j] + B3 (B3 the same for all phases: only centres the logits).
All four phases go through the identical increasing map V -> logit, so argmax(logits) == argmax(V)
(the untouched random W3 columns, ~1e-3, only matter at exact ties).
Units 12..63 of both layers keep their random initial weights and their random initial W3 columns.

Run:  python work/build.py   (from the pack folder)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import tools  # noqa: E402
import rule   # noqa: E402

UNITS = [(0.30, 0.0), (0.30, 1.2), (0.08, 1.2)]  # (gain 1/vehicle, offset) per layer-1 copy
K2 = 1.2                                          # layer-2 gain
BETA = 4.25                                       # output scale


def profile(v, units=UNITS, k2=K2, beta=BETA):
    """logit as a function of the vehicle score V (before the common shift B3)."""
    v = np.asarray(v, dtype=np.float64)
    return beta * sum(np.tanh(k2 * np.tanh(a * v - c)) for a, c in units)


# common shift so that logits are centred between V = 0 and V = large (does not change probabilities)
B3 = -0.5 * (profile(0.0) + profile(1e3))


def build(units=UNITS, k2=K2, beta=BETA, b3=B3):
    w = tools.load("start_weights.npz")
    A = rule.coef_matrix()
    n = len(units)
    nd = 4 * n
    for p in range(4):
        for j, (a, c) in enumerate(units):
            u = n * p + j
            w["W1"][u, :] = a * A[p]
            w["b1"][u] = -c
    w["W2"][:nd, :] = 0.0
    w["b2"][:nd] = 0.0
    for u in range(nd):
        w["W2"][u, u] = k2
    w["W3"][:, :nd] = 0.0
    for p in range(4):
        for j in range(n):
            w["W3"][p, n * p + j] = beta
    w["b3"][:] = b3
    return w


if __name__ == "__main__":
    w = build()
    out = os.path.join(HERE, "weights.npz")
    tools.save(w, out)
    print("saved", out, " B3 =", round(float(B3), 3))
    print("logit profile V=0..20:", np.round(profile(np.arange(21)) + B3, 2))
    print(tools.trainability(tools.load(out)))
    print("edit size vs start:", tools.edit_size(tools.load(out), tools.load("start_weights.npz")))
