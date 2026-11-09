"""Fuller readable translation of start_weights.npz.

Same structure as rules_small.py, with the details that the network also uses:
  * EW lanes split into the straight lanes (E1, W1) and the shared right+straight lanes (E0, W0),
    because the network treats them differently when it weighs EW-left and NS-left;
  * the upstream lanes that carry left-turners (lane 1 upstream) add a little to the left-turn phases.

Once the 10 s minimum green has elapsed the network switches to the best-scoring other phase
(score + phase-transition bonus); it holds EW-left only when a left pocket is still occupied and the
best alternative is weak.

Lane load = vehicles on the lane + halting vehicles on it (a stopped car counts 2, a moving car 1);
stop-line lanes hold about 3 cars, so a lane load is 0..6. Upstream counts are vehicles (0..34).
"""
import numpy as np

LANES = ["N0", "N1", "N2", "E0", "E1", "E2", "S0", "S1", "S2", "W0", "W1", "W2"]
UP = ["N0u", "N1u", "E0u", "E1u", "S0u", "S1u", "W0u", "W1u"]
UP_CAPACITY = 256.0 / 7.5   # vehicles an upstream lane can hold

# Phase-transition bonus BONUS[current][candidate], in lane-load units.
BONUS = {
    0: {1: 3.7, 2: -1.05, 3: -2.65},   # after NS-through: NS-left strongly favoured, EW-left disfavoured
    1: {0: 1.0, 2: -0.6, 3: -0.4},
    2: {0: 0.6, 1: 0.2, 3: -0.8},
    3: {0: 2.75, 1: -0.95, 2: -1.8},   # after EW-left: NS-through favoured
}


def decide(x):
    x = np.asarray(x, dtype=float)
    ld = {n: 3.0 * (x[i] + x[12 + i]) for i, n in enumerate(LANES)}
    up = {n: UP_CAPACITY * x[32 + k] for k, n in enumerate(UP)}
    current = int(np.argmax(x[40:44]))

    ns_thru = ld["N1"] + 0.75 * (ld["S0"] + ld["S1"]) + 0.25 * ld["N0"]
    ns_left = ld["N2"] + ld["S2"]
    ew_str = ld["E1"] + 1.5 * ld["W1"]          # EW straight-only lanes
    ew_rgt = ld["E0"] + ld["W0"]                # EW right+straight lanes
    ew_left = ld["E2"] + ld["W2"]
    up_ns = up["N1u"] + up["S1u"]               # NS upstream lanes feeding straight + left
    up_ew = up["E0u"] + up["E1u"] + up["W0u"] + up["W1u"]

    score = [
        # 0 NS-through
        ns_thru - 0.45 * ew_str + 0.2 * ew_left + 0.1 * up_ns,
        # 1 NS-left
        1.35 * ns_left - 0.3 * ew_rgt - 0.1 * ew_left + 0.15 * up_ns,
        # 2 EW-through
        1.1 * ew_str + 0.75 * ew_rgt + 0.5 * ns_thru - 0.55 * ns_left + 0.1 * up_ew,
        # 3 EW-left
        1.55 * ew_left + 0.5 * ew_rgt - 0.2 * ew_str - 0.25 * ns_thru - 0.15 * ns_left + 0.2 * up_ew,
    ]
    for j in BONUS[current]:
        score[j] += BONUS[current][j]
    score[current] = -1e9

    best = 0
    if score[1] > score[best]:
        best = 1
    if score[2] > score[best]:
        best = 2
    if score[3] > score[best]:
        best = 3

    # hold EW-left while its pockets still hold a car and the best alternative is weak
    if current == 3 and ew_left >= 1 and score[best] < 3.5 + 2.5 * ew_left:
        return 3
    return best
