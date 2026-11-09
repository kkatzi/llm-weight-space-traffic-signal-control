"""v6 = v3 design with terms 2-4 on the top-44 units, plus an N1 fix: a halted car on N1 also drives the
q_N0/d_N0 columns at half strength (W1[:,1] += 0.5*(W1[:,0] + W1[:,12])), because the original network ignores
a lone N1 queue and would strand it forever."""
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "work"))
import tools
from build import build, idle_table
w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
w = build(c_t=0.6, k3=0.25, k_sp=3, topk=44)
w["W1"][:, 1] += 0.5 * (w0["W1"][:, 0] + w0["W1"][:, 12])
es = tools.edit_size(w, w0); print(es); assert es["changed"] <= 371
print(idle_table(w))
tools.save(w, os.path.join(HERE, "work", "v6_n1fix.npz"))
