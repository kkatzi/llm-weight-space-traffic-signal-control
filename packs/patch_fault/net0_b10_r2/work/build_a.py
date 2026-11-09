"""Candidate A: impute the dead W2 pocket detector from its mirror, the E2 pocket.

Since x[11] (q_W2) and x[23] (d_W2) are always 0, their W1 columns do nothing. Substituting q_W2 := q_E2 and
d_W2 := d_E2 inside the first layer is exact by linearity:
    W1[:, 5]  += W1[:, 11]      (q_E2 now also carries the q_W2 weights)
    W1[:, 17] += W1[:, 23]      (d_E2 now also carries the d_W2 weights)
128 parameters change. Columns 11 and 23 are left untouched (they multiply 0).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools  # noqa: E402

w0 = tools.load("start_weights.npz")
w = {k: v.copy() for k, v in w0.items()}
w["W1"][:, 5] += w0["W1"][:, 11]
w["W1"][:, 17] += w0["W1"][:, 23]
print(tools.edit_size(w, w0))
tools.save(w, os.path.join(HERE, "cand_a.npz"))
