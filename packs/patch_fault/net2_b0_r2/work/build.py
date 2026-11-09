"""Build work/weights.npz from start_weights.npz by construction (no fitting).

Edit 1 - impute the dead W2 detector from its mirror lane E2 (same demand, served by the same phase 3):
    x[11] (q_W2) := x[5] (q_E2),  x[23] (d_W2) := x[17] (d_E2)
  folded into layer 1:  W1[:, 5] += W1[:, 11];  W1[:, 17] += W1[:, 23]          (128 parameters)

Edit 2 - repurpose layer-1 unit J (the unit whose mean-ablation changes the fewest honoured decisions) into a
  "W2 pocket is spilling back" detector:
    s = tanh(A * q_W1u - P * phase3 - D)   (~ +1 when >= 1 halting vehicle on W1u and phase 3 is not green)
  Mirror evidence from the E side: when q_E1u > 0 and phase 3 is not green, the E2 pocket is full in >= 93 % of
  recorded start-network / actuated states. So s = +1 means "W2 is full: q_W2 = d_W2 = 1".
  Its outgoing weights W2[:, J] carry the average layer-2 pre-activation shift that the ORIGINAL network shows when
  (q_W2, d_W2) goes from the imputed (q_E2, d_E2) to (1, 1), on recorded spillback states (forward passes only).
  The old unit J is replaced by its mean activation, compensated in b2.
    W1[J, :] (46) + b1[J] (1) + W2[:, J] (64) + b2 (64)                           (175 parameters)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools  # noqa: E402

I_QE2, I_DE2, I_QW2, I_DW2 = 5, 17, 11, 23
I_QW1U, I_PH3, I_OK = 31, 43, 44
J = 50                       # layer-1 unit to repurpose (see analyse_units.py)
A, P, D = 120.0, 130.0, 1.75  # detector: 1 halting vehicle on W1u = 1/34.13 = 0.0293 -> pre-act +1.77

w0 = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64)
ph = X[:, 40:44].argmax(1)

# ---- edit 1: copy imputation
w = {k: v.copy() for k, v in w0.items()}
w["W1"][:, I_QE2] += w0["W1"][:, I_QW2]
w["W1"][:, I_DE2] += w0["W1"][:, I_DW2]

# ---- edit 2: spillback detector on unit J
h1_copy = np.tanh(X @ w["W1"].T + w["b1"])
mean_J = h1_copy[:, J].mean()

# average layer-2 pre-activation shift of the ORIGINAL network when W2 goes from (q_E2, d_E2) to (1, 1),
# over recorded states where the detector would fire and the E2 pocket is not already full
spill = (ph != 3) & (X[:, I_QW1U] > 0.01) & (X[:, I_QE2] < 0.99)
Xc = X[spill].copy()
Xc[:, I_QW2] = Xc[:, I_QE2]
Xc[:, I_DW2] = Xc[:, I_DE2]
Xf = Xc.copy()
Xf[:, I_QW2] = 1.0
Xf[:, I_DW2] = 1.0
a2c = np.tanh(Xc @ w0["W1"].T + w0["b1"]) @ w0["W2"].T
a2f = np.tanh(Xf @ w0["W1"].T + w0["b1"]) @ w0["W2"].T
v = (a2f - a2c).mean(0)

s_off = np.tanh(-D)
s_on = np.tanh(A / 34.13 - D)
w["W1"][J, :] = 0.0
w["W1"][J, I_QW1U] = A
w["W1"][J, I_PH3] = -P
w["b1"][J] = -D
w["W2"][:, J] = v / (s_on - s_off)
w["b2"] = w0["b2"] + w0["W2"][:, J] * mean_J - w["W2"][:, J] * s_off

out = os.path.join(HERE, "weights.npz")
tools.save(w, out)
info = {"unit": J, "A": A, "P": P, "D": D, "s_off": float(s_off), "s_on_1veh": float(s_on),
        "mean_J_removed": float(mean_J), "n_spill_states_for_v": int(spill.sum()),
        "edit_size": tools.edit_size(tools.load(out), w0)}
print(json.dumps(info, indent=1))
