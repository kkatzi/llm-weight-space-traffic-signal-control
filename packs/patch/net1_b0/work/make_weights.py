"""Final construction: build the edited network, add partial mean-compensation in b2 with the leftover budget,
save work/weights.npz. Everything is set by construction; forward passes on recorded states are used only to
read mean activations (for bias compensation) and to check the result."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, tools
from build import build, ideal_logits

CFG = dict(alpha=0.75, M=2.5, G=5.0, eps=0.01)
K_UNITS, J_UNITS = (31, 5, 27), (25, 46, 4)
LIMIT = 371

w0 = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); ok = X[:, 44] > .5
w = build(w0, X_mean=X[ok], k_units=K_UNITS, j_units=J_UNITS, **CFG)
# compensate the lost mean input of the repurposed layer-1 units on the layer-2 units with the largest shift
h1, _ = tools.hidden(w0, X[ok])
shift = sum(w0["W2"][:, k] * h1[:, k].mean() for k in K_UNITS)
shift[list(J_UNITS)] = 0.0
budget = LIMIT - tools.edit_size(w, w0)["changed"]
for i in np.argsort(-np.abs(shift))[:budget]:
    w["b2"][i] += shift[i]
es = tools.edit_size(w, w0)
assert es["changed"] <= LIMIT, es
tools.save(w, os.path.join(HERE, "weights.npz"))
print("saved; edit size", es)
