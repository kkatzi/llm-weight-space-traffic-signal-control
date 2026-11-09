"""Candidate D8: candidate D with a stronger spillback weight (q_W1u coefficient 8 instead of 5)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_d as D  # noqa: E402
import impute as I  # noqa: E402
import tools  # noqa: E402

D.G = 8.0
w = D.make()
print(tools.edit_size(w, I.W0))
tools.save(w, os.path.join(HERE, "cand_d8.npz"))
