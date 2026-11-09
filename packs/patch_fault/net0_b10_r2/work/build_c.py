"""Candidate C: like B, but the E2 mirror enters as a conditional-mean estimate (0.15 + 0.6 * E2) instead of a copy.

    q_W2_hat = 0.6 * q_E2 + 5 * q_W1u + 1.0 * t_green + (0.15 - 0.25) * ok
    d_W2_hat = 0.6 * d_E2 + 5 * q_W1u + 1.0 * t_green + (0.15 - 0.25) * ok
(the ok input is 1 in every state whose decision takes effect, so its coefficient acts as an intercept)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import impute as I  # noqa: E402
import tools  # noqa: E402

e, b0, g, a, c = 0.6, 0.15, 5.0, 1.0, 0.25
cq = {"q_E2": e, "q_W1u": g, "t_green": a, "ok": -a * c + b0}
cd = {"d_E2": e, "q_W1u": g, "t_green": a, "ok": -a * c + b0}
w = I.build(cq, cd)
print(tools.edit_size(w, I.W0))
tools.save(w, os.path.join(HERE, "cand_c.npz"))
