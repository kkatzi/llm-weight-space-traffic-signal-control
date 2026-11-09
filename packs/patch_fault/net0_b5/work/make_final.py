"""Build the delivered weights (design 'E8') from start_weights.npz and save work/weights.npz.

All numbers are hand-set (see NOTES.md). Only five first-layer columns and b1 change (5 x 64 + 64 = 320 values):
  b1, W1[:, q_E2], W1[:, q_W1u], W1[:, t_green], W1[:, phase3]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools  # noqa: E402

I = {k: i for i, k in enumerate(tools.names())}
QW2, DW2 = I["q_W2"], I["d_W2"]

# 1) estimate of the dead W2 inputs, as a linear function of working inputs
#    q_hat = 0.04 + 0.8 q_E2 + 3 q_W1u + 0.4 t_green - 0.2 phase3
#    d_hat = 0.12 + 0.8 q_E2 + 3 q_W1u + 0.4 t_green - 0.2 phase3
EST_Q = {"const": 0.04, "q_E2": 0.8, "q_W1u": 3.0, "t_green": 0.4, "phase3": -0.2}
EST_D = {"const": 0.12, "q_E2": 0.8, "q_W1u": 3.0, "t_green": 0.4, "phase3": -0.2}
# 2) lone-car timer: phantom N/S through demand 0.4 * (t_green - 0.3) on q and d of N0, N1, S0, S1
NS_TIMER, NS_CENTER = 0.4, 0.3
NS = ["q_N0", "q_N1", "q_S0", "q_S1", "d_N0", "d_N1", "d_S0", "d_S1"]


def build(w0):
    w = {k: v.copy() for k, v in w0.items()}
    W1 = w0["W1"]
    for name in set(EST_Q) | set(EST_D):
        v = W1[:, QW2] * EST_Q.get(name, 0.0) + W1[:, DW2] * EST_D.get(name, 0.0)
        if name == "const":
            w["b1"] = w["b1"] + v
        else:
            w["W1"][:, I[name]] = w["W1"][:, I[name]] + v
    for nm in NS:
        w["W1"][:, I["t_green"]] += NS_TIMER * W1[:, I[nm]]
        w["b1"] -= NS_TIMER * NS_CENTER * W1[:, I[nm]]
    return w


if __name__ == "__main__":
    w0 = tools.load("start_weights.npz")
    w = build(w0)
    out = os.path.join(HERE, "work", "weights.npz")
    tools.save(w, out)
    wf = tools.load(out)
    print("edit size:", tools.edit_size(wf, w0))
    e8 = os.path.join(HERE, "work", "w_E8.npz")
    if os.path.exists(e8):
        we8 = tools.load(e8)
        print("identical to evaluated w_E8.npz:", all(np.array_equal(wf[k], we8[k]) for k in wf))
