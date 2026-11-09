"""v1: symmetric-twin substitution for the failed W2 detector.

The failed inputs x[11] (q_W2) and x[23] (d_W2) are always 0. The E left-turn pocket E2 has the same demand
(E and W share the same demand and directional bias, 30 % left) and is served by the same phase (3 EW_left),
so its readings are the natural stand-in. We let every first-layer unit read q_E2 in place of q_W2 and d_E2 in
place of d_W2 by adding the W2 weight columns onto the E2 columns:
    W1[:, q_E2] += W1[:, q_W2];  W1[:, d_E2] += W1[:, d_W2]
(the W2 columns themselves are left as they are; they multiply 0).
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools  # noqa: E402

QW2, DW2, QE2, DE2 = 11, 23, 5, 17


def build():
    w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
    w = {k: v.copy() for k, v in w0.items()}
    w["W1"][:, QE2] += w0["W1"][:, QW2]
    w["W1"][:, DE2] += w0["W1"][:, DW2]
    return w0, w


if __name__ == "__main__":
    w0, w = build()
    print(tools.edit_size(w, w0))
    tools.save(w, os.path.join(HERE, "work", "v1_twin.npz"))
