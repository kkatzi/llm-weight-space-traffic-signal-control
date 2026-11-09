"""Build the repaired network by construction (no fitting).

The dead detector W2 feeds x[11] (q_W2) and x[23] (d_W2), which are always 0 now. We substitute a
hand-built estimate of each, as a linear function of inputs that still work:

    q_W2_hat = q_E2 + A * t_green - G * phase3 (+ C * q_W1u)
    d_W2_hat = d_E2 + A * t_green - G * phase3 (+ C * q_W1u)

Because layer 1 is linear in x before the tanh, "feeding x[11] := q_W2_hat" is the same as adding
W1[:, 11] * coef to the W1 column of each input that appears in q_W2_hat (and likewise W1[:, 23] for
d_W2_hat). Only W1 columns change: 64 parameters per input column used.

Reasoning for each term:
  * q_E2 / d_E2 (mirror): the E and W left pockets are served by the same phase (3), have the same
    demand, so their fill follows the same red-time clock. This is what the network saw in training,
    when E2 and W2 moved together.
  * A * t_green: outside phase 3, the W2 pocket has been red for at least the current green time.
    Left-turn arrivals from W are ~225 veh/h nominal, i.e. the 3-car pocket fills in about 48 s, so
    the fill lower bound grows ~1.25 per 60 s (= per unit of t_green). This term is what breaks the
    end-of-episode deadlock (W2 holds cars, everything else is empty, the faulty net holds phase 2
    forever). A is set below 1.25 to limit double counting with the mirror term.
  * -G * phase3: in phase 3 the W2 pocket is being served, so the time term should not apply.
    G = 0.2 * A cancels the time term exactly at the first honoured decision of phase 3 (12 s, t_green
    = 0.2), so phase 3 is judged on the mirror alone, as before, and cannot lock itself in.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import tools  # noqa: E402

Q_W2, D_W2 = 11, 23
Q_E2, D_E2 = 5, 17
PHASE3, T_GREEN, Q_W1U = 43, 45, 31


def build(A=0.5, G=None, C=0.0, mirror=1.0):
    G = 0.2 * A if G is None else G
    w0 = tools.load("start_weights.npz")
    w = {k: v.copy() for k, v in w0.items()}
    W1 = w0["W1"]
    # (target input column, coefficient for q_W2_hat, coefficient for d_W2_hat)
    terms = [(Q_E2, mirror, 0.0), (D_E2, 0.0, mirror), (T_GREEN, A, A), (PHASE3, -G, -G)]
    if C:
        terms.append((Q_W1U, C, C))
    for j, cq, cd in terms:
        w["W1"][:, j] = W1[:, j] + cq * W1[:, Q_W2] + cd * W1[:, D_W2]
    return w


if __name__ == "__main__":
    cfg = dict(A=0.5, G=0.1, C=0.0, mirror=1.0)
    w = build(**cfg)
    w0 = tools.load("start_weights.npz")
    print(tools.edit_size(w, w0))
    out = os.path.join(HERE, "weights.npz")
    tools.save(w, out)
    with open(os.path.join(HERE, "build_config.json"), "w") as f:
        json.dump(cfg, f)
    print("saved", out)
