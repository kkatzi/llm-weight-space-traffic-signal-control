"""Candidate v7 (NOT the delivered weights; evaluated, rejected): W2-pocket estimate fed into the dead W2 columns + N1 lone-car fix.

    W2hat = E2 + 0.6*(t_green - 0.2) - 0.25*phase3 + 3*q_W1u      (q and d channels; see build.py)
    q_N1 additionally drives the q_N0/d_N0 columns at 0.6 strength  (W1[:,1] += 0.6*(W1[:,0] + W1[:,12]))
Terms other than E2 are written on the 44 units with the largest |W1[:,11] + W1[:,23]| to stay within 371 edits."""
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "work"))
import tools
from build import build, idle_table
w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
w = build(c_t=0.6, t0=0.2, k3=0.25, k_sp=3, topk=44)
w["W1"][:, 1] += 0.6 * (w0["W1"][:, 0] + w0["W1"][:, 12])
es = tools.edit_size(w, w0); print(es); assert es["changed"] <= 371
print(idle_table(w))
out = sys.argv[1] if len(sys.argv) > 1 else "v7_final.npz"
tools.save(w, os.path.join(HERE, "work", out))
