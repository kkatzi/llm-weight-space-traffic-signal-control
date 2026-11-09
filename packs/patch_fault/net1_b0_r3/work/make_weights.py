"""Build work/weights.npz: route the dead W2-pocket inputs onto the E2-pocket inputs (mirror imputation).

The failed detector makes x[11] (q_W2) and x[23] (d_W2) always 0, so W1[:, 11] and W1[:, 23] never contribute.
E and W carry the same demand and their left pockets are served together by phase 3, so the best single stand-in for
the west pocket is the east pocket. Feeding q_E2 / d_E2 through the W2 columns as well is exactly the same as
evaluating the original network on an observation where q_W2 := q_E2 and d_W2 := d_E2:

    W1[:, 5]  += W1[:, 11]      (q_E2 column also carries the q_W2 weights)
    W1[:, 17] += W1[:, 23]      (d_E2 column also carries the d_W2 weights)

Nothing is fitted: the change is fixed by the symmetry argument. 128 parameters change.
"""
import os
from common import *

w = copyw(W0)
w["W1"][:, 5] = W0["W1"][:, 5] + W0["W1"][:, 11]
w["W1"][:, 17] = W0["W1"][:, 17] + W0["W1"][:, 23]

out = os.path.join(HERE, "weights.npz")
tools.save(w, out)
w_saved = tools.load(out)
print("edit size", tools.edit_size(w_saved, W0))

# exactness: patched net on the faulty observations == original net on mirror-imputed observations
lp = tools.forward(w_saved, X)
lo = tools.forward(W0, impute(X))
print("max |logit diff| patched vs original-on-imputed:", np.abs(lp - lo).max(),
      " argmax agreement:", (lp.argmax(1) == lo.argmax(1)).mean())
