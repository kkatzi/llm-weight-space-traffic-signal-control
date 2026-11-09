"""Candidate B: A (mirror E2 -> W2) + spillback evidence from W1u + long-green prior.

    q_W2_hat = q_E2 + g * q_W1u + a * (t_green - c * ok)
    d_W2_hat = d_E2 + g * q_W1u + a * (t_green - c * ok)
Four live columns (q_E2, d_E2, q_W1u, t_green, ok -> 5 columns... q_E2 and d_E2 each take one) = 5 x 64 = 320.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import impute as I  # noqa: E402
import tools  # noqa: E402

g, a, c = 5.0, 1.0, 0.25
cq = {"q_E2": 1.0, "q_W1u": g, "t_green": a, "ok": -a * c}
cd = {"d_E2": 1.0, "q_W1u": g, "t_green": a, "ok": -a * c}
w = I.build(cq, cd)
print(tools.edit_size(w, I.W0))
tools.save(w, os.path.join(HERE, "cand_b.npz"))
