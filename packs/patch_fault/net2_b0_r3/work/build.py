"""Build the repaired network by construction (no fitting).

Step 1 (W2 imputation): the failed inputs q_W2 (x11) and d_W2 (x23) are replaced, inside layer 1, by an estimate
built from the opposing left pocket E2 (same demand by symmetry, served by the same phase 3):
    W2_hat = RHO * E2 + (1 - RHO) * mu(phase),   mu = mean of E2 in phases 0-2, or in phase 3
Since x11 = x23 = 0 at test time, W1[:,11] * W2_hat is folded into the columns of E2 (x5, x17), the bias b1 and
the phase-3 one-hot column (x43). Columns 11 and 23 are left unchanged (they multiply zeros).
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools, numpy as np

RHO = 0.8
# E2 statistics on the start-network recorded states (computed in stats_e2.py)
MU_Q_OTH, MU_D_OTH = 0.407, 0.457   # phases 0-2
MU_Q_P3, MU_D_P3 = 0.056, 0.219     # phase 3


def impute_w2(w0, rho=RHO, mq=MU_Q_OTH, md=MU_D_OTH, mq3=MU_Q_P3, md3=MU_D_P3):
    w = {k: v.copy() for k, v in w0.items()}
    cq, cd = w0["W1"][:, 11].copy(), w0["W1"][:, 23].copy()
    w["W1"][:, 5] += rho * cq
    w["W1"][:, 17] += rho * cd
    w["b1"] += (1 - rho) * (cq * mq + cd * md)
    w["W1"][:, 43] += (1 - rho) * (cq * (mq3 - mq) + cd * (md3 - md))
    return w


if __name__ == "__main__":
    w0 = tools.load("start_weights.npz")
    w = impute_w2(w0)
    print(tools.edit_size(w, w0))
