"""Small readable translation of start_weights.npz (at most 8 comparisons; this file has 6).

How the network decides, once the 10 s minimum green has elapsed:
  * it (almost) never extends the current green: it switches to one of the other three phases;
  * it gives every phase a demand score built from the stop-line lanes and picks the highest
    score plus a phase-transition bonus (which phase may follow which);
  * one exception: in EW-left with a car still in a left pocket and little demand elsewhere, it holds EW-left.

Lane load = vehicles on the lane + halting vehicles on it (a stopped car counts 2, a moving car 1);
each stop-line lane holds about 3 cars, so a lane load is between 0 and 6. Upstream counts are vehicles.
"""
import numpy as np

LANES = ["N0", "N1", "N2", "E0", "E1", "E2", "S0", "S1", "S2", "W0", "W1", "W2"]
UP = ["N0u", "N1u", "E0u", "E1u", "S0u", "S1u", "W0u", "W1u"]
UP_CAPACITY = 256.0 / 7.5   # vehicles an upstream lane can hold

# Phase-transition bonus: BONUS[current][candidate], in lane-load units.
# The current phase is never a candidate (the network does not extend greens).
BONUS = {
    0: {1: 3.0, 2: -1.05, 3: -1.95},   # after NS-through: NS-left is favoured
    1: {0: 1.1, 2: -0.7, 3: -0.45},
    2: {0: 0.65, 1: 0.0, 3: -0.6},     # after EW-through: mild preference for NS-through
    3: {0: 2.8, 1: -1.1, 2: -1.7},     # after EW-left: NS-through is favoured
}


def decide(x):
    x = np.asarray(x, dtype=float)
    ld = {n: 3.0 * (x[i] + x[12 + i]) for i, n in enumerate(LANES)}
    up = {n: UP_CAPACITY * x[32 + k] for k, n in enumerate(UP)}
    current = int(np.argmax(x[40:44]))

    # demand of each movement group (lane loads; lane 0 = right+straight, 1 = straight, 2 = left pocket)
    ns_thru = ld["N1"] + 0.75 * (ld["S0"] + ld["S1"]) + 0.25 * ld["N0"]
    ns_left = ld["N2"] + ld["S2"]
    ew_thru = ld["E1"] + 1.5 * ld["W1"] + 0.75 * (ld["E0"] + ld["W0"])
    ew_left = ld["E2"] + ld["W2"]
    up_ns = up["N1u"] + up["S1u"]                          # vehicles approaching on NS upstream lanes 1
    up_ew = up["E0u"] + up["E1u"] + up["W0u"] + up["W1u"]  # vehicles approaching on EW upstream lanes

    # phase scores (main term = own demand; small cross terms copy the network's interactions)
    score = [
        ns_thru + 0.2 * ew_left + 0.1 * up_ns,                                        # 0 NS-through
        1.35 * ns_left + 0.15 * up_ns,                                                # 1 NS-left
        1.25 * ew_thru + 0.5 * ns_thru - 0.55 * ns_left + 0.1 * up_ew,                # 2 EW-through
        1.55 * ew_left + 0.45 * ew_thru - 0.25 * ns_thru - 0.15 * ns_left + 0.2 * up_ew,  # 3 EW-left
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
    if current == 3 and ew_left >= 1 and score[best] < 5.0 + 3.0 * ew_left:
        return 3
    return best
