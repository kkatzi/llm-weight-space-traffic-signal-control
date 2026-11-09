"""Full repair = step 1 (W2 imputation, build.py) + step 2 (two max-green rule pathways, pathways.py).

Pathway A: phase 2 (EW_through) green >= 60 s            -> push the phase-2 logit down (leave phase 2).
Pathway B: phase 3 (EW_left) green >= 27 s and E2 empty   -> push phase 2 up and phase 3 down (go to phase 2).
Both only fire in near-empty 'stuck' situations (drain-out deadlocks); in the recorded demand periods phase 2
never lasted 60 s and phase 3 rarely 27 s with E2 empty (5 of 5116 states).
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools, numpy as np
from build import impute_w2
from pathways import add_pathway

K = 400.0        # sharpness of the detector conditions (worst-case margins checked in bounds.py)
KE = 2.5 * K     # 'E2 empty' weight in B: one car (0.335) must beat the t_green ramp (at most 0.592 K)
K2 = 20.0        # relay weight layer 1 -> layer 2 (old layer-2 pre-activations stay within about +-12)
L = 6.0          # logit push (normal -> condition difference is 2L = 12)
THETA_A = 0.975  # phase 2 green >= 60 s (decision values ... 0.95, 1.0)
THETA_B = 0.408  # phase 3 green >= 27 s (decision values ... 0.367, 0.45)
U_A, J_A = 50, 1
U_B, J_B = 38, 63
U_B_NORMAL = -1.0


def build(w0, X, with_A=True, with_B=True):
    w = impute_w2(w0)
    if with_A:
        # A: margin = x45 + x42 - 1 - THETA_A  (> 0 only in phase 2 with t_green = 1)
        w = add_pathway(w, X, U_A, +1.0, J_A, {45: K, 42: K}, -K * (1 + THETA_A),
                        P=np.array([0.0, 0.0, -L, 0.0]), K2=K2)
    if with_B:
        # B: margin = x45 + x43 - 1 - THETA_B - 2.5 (x5 + x17)  (> 0 only in phase 3, t_green >= 0.45, E2 empty)
        w = add_pathway(w, X, U_B, U_B_NORMAL, J_B, {45: K, 43: K, 5: -KE, 17: -KE}, -K * (1 + THETA_B),
                        P=np.array([0.0, 0.0, +L, -L]), K2=K2)
    return w


if __name__ == "__main__":
    w0 = tools.load("start_weights.npz")
    X = tools.states()["obs"].astype(np.float64)
    w = build(w0, X)
    print(tools.edit_size(w, w0))
