"""v1: route the dead W2 detector's weights onto the E2 detector (E/W symmetry).

x[11]=q_W2 and x[23]=d_W2 are always 0, so their W1 columns do nothing. The east left pocket E2 has the same demand
(750 veh/h x 30 % left) and is served by the same phase (3, EW_left), so q_E2 / d_E2 is the natural stand-in.
Adding W1[:,11] to W1[:,5] and W1[:,23] to W1[:,17] makes every hidden unit see q_W2 := q_E2, d_W2 := d_E2.
Changes: 64 + 64 = 128 parameters.
"""
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools

w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
w = {k: v.copy() for k, v in w0.items()}
w["W1"][:, 5] += w0["W1"][:, 11]
w["W1"][:, 17] += w0["W1"][:, 23]
print(tools.edit_size(w, w0))
tools.save(w, os.path.join(HERE, "work", "v1.npz"))
