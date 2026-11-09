"""v1: feed the opposite left pocket (E2) into the dead W2 input columns.
q_W2 := q_E2, d_W2 := d_E2  (same phase EW_left serves both; E and W demand symmetric)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
w = tools.load("start_weights.npz")
w0 = tools.load("start_weights.npz")
w["W1"][:, 5] += w0["W1"][:, 11]
w["W1"][:, 17] += w0["W1"][:, 23]
print(tools.edit_size(w, w0))
tools.save(w, "work/v1_proxyE2.npz")
