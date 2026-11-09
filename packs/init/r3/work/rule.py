"""Reference rule (plain numpy, no weights): per-phase vehicle score, used to design and check the network.

For each phase p (vehicle units):
  V_p = sum over p's stop-line lanes of CAP_S*(WS_Q*q + WS_D*d)          (halting count 1, moving count 0.5)
      + sum over p's upstream lanes  share*CAP_U*(WU_Q*q + WU_D*d)      (halting 1, moving 0.1)
      + B0 * [p is the current phase]                                   (switching costs 3 s yellow + start-up)
decision = argmax_p V_p

Upstream lane X0u feeds only the through phase (share 1). Upstream lane X1u feeds both the through lane X1 and
the left pocket X2; in the recorded data a queue on X1u mostly comes with a full left pocket (left-turn
spillback), so its share is 0.35 through / 0.65 left.
"""
import numpy as np

CAP_S = 22.4 / 7.5   # 2.987 vehicles per stop-line lane
CAP_U = 256.0 / 7.5  # 34.13 vehicles per upstream lane

WS_Q, WS_D = 0.5, 0.5     # stop line: 3*(0.5 q + 0.5 d)  -> halting vehicle 1.0, moving vehicle 0.5
WU_Q, WU_D = 0.9, 0.1     # upstream: 34*(0.9 q + 0.1 d)  -> halting vehicle 1.0, moving vehicle 0.1
B0 = 0.6                  # current-phase bonus, vehicles (< 1 so that a single halting car always wins a switch)
SH_T, SH_L = 0.35, 0.65   # split of the shared upstream lane X1u between through and left

# stop-line lane indices per phase (into the 12 queue / 12 density entries)
STOP = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
# upstream lanes (index 0..7 within the 8 upstream lanes: N0u N1u E0u E1u S0u S1u W0u W1u) and share per phase
UP = {0: {0: 1.0, 1: SH_T, 4: 1.0, 5: SH_T},
      1: {1: SH_L, 5: SH_L},
      2: {2: 1.0, 3: SH_T, 6: 1.0, 7: SH_T},
      3: {3: SH_L, 7: SH_L}}


def coef_matrix(b0=B0, ws=(WS_Q, WS_D), wu=(WU_Q, WU_D)):
    """A (4, 46) with V = X @ A.T (the bonus sits on the phase one-hot inputs 40..43)."""
    A = np.zeros((4, 46))
    for p in range(4):
        for i in STOP[p]:
            A[p, i] += CAP_S * ws[0]
            A[p, 12 + i] += CAP_S * ws[1]
        for u, sh in UP[p].items():
            A[p, 24 + u] += sh * CAP_U * wu[0]
            A[p, 32 + u] += sh * CAP_U * wu[1]
        A[p, 40 + p] = b0
    return A


def scores(X, **kw):
    return np.atleast_2d(np.asarray(X, dtype=np.float64)) @ coef_matrix(**kw).T


def decide(x):
    return int(scores(x).argmax())
