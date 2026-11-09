"""Candidate D: W2 imputation (mirror E2 + W1u spillback) + two long-green priors anchored at 12 s of green.

First layer sees, in place of the dead detector,
    q_W2_hat = q_E2 + 5 * q_W1u + 1.0 * (t_green - 0.2 * ok)
    d_W2_hat = d_E2 + 5 * q_W1u + 1.0 * (t_green - 0.2 * ok)
and, on top of the live NS-through inputs (q/d of N0, N1, S0, S1), a phantom
    + 0.5 * (t_green - 0.2 * ok)            per lane and channel
so that after a long green (rest with nothing visible) the network both visits phase 3 (hidden W2 cars) and
serves a lone NS-through car it would otherwise ignore. At t_green = 0.2 (12 s, the first decision that can
switch and by far the most common one) both priors are exactly zero.
Changed columns of W1: q_E2, d_E2, q_W1u, t_green, ok -> 5 x 64 = 320 parameters.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import impute as I  # noqa: E402
import tools  # noqa: E402

E, G, A, C = 1.0, 5.0, 1.0, 0.2          # mirror weight, spillback weight, W2 prior slope, prior anchor
B, CB = 2.0, 0.2                          # NS-through phantom: total slope over the 4 lanes, anchor
NS = ("N0", "N1", "S0", "S1")


def make():
    w = I.build({"q_E2": E, "q_W1u": G, "t_green": A, "ok": -A * C},
                {"d_E2": E, "q_W1u": G, "t_green": A, "ok": -A * C})
    W1_0 = I.W0["W1"]
    for ln in NS:
        for pre in ("q_", "d_"):
            j = I.IDX[pre + ln]
            w["W1"][:, I.IDX["t_green"]] += B / len(NS) * W1_0[:, j]
            w["W1"][:, I.IDX["ok"]] += -B * CB / len(NS) * W1_0[:, j]
    return w


if __name__ == "__main__":
    w = make()
    print(tools.edit_size(w, I.W0))
    tools.save(w, os.path.join(HERE, "cand_d.npz"))
