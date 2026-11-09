"""Build the repaired network by construction (no fitting).

Fault: x[11] = q_W2 and x[23] = d_W2 are always 0.
Repair: feed the partner pocket E2 (served by the same phase 3, same demand as W2) into the inputs the network
used for W2.  For every layer-1 unit j:
    W1[j, 5]  += W1[j, 11]     (q_E2 now also plays the role of q_W2)
    W1[j, 17] += W1[j, 23]     (d_E2 now also plays the role of d_W2)
With the failed detector this computes exactly what the original network computes on an input where
q_W2 := q_E2 and d_W2 := d_E2.  128 parameters change; nothing else is touched.
"""
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np, tools

w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
w = {k: v.copy() for k, v in w0.items()}
Q_E2, Q_W2, D_E2, D_W2 = 5, 11, 17, 23
w["W1"][:, Q_E2] += w0["W1"][:, Q_W2]
w["W1"][:, D_E2] += w0["W1"][:, D_W2]

out = os.path.join(HERE, "work", "weights.npz")
tools.save(w, out)
print("saved", out)
print("edit size", tools.edit_size(tools.load(out), w0))
