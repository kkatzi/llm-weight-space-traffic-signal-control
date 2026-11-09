"""rules_full.py: readable translation of start_weights.npz (PPO actor, junction J0), more detailed version.

Same structure as rules_small.py: when the minimum green has elapsed, hand the green to the phase with the
highest score; the score is the phase's own weighted queue, plus small spill-over credits the network gives
through phases for traffic elsewhere, plus a preference that depends on the phase that is green now.

Differences from rules_small.py: lane-by-lane weights inside each movement (the network weighs S2 more than N2,
E1 more than E0, S1 less than N0/S0), the cross credits, and upstream N/S traffic.

Units: "vehicles". On a stop-line lane a halted vehicle counts 1 and a moving one 0.5 (lane holds 3 vehicles).
On an upstream lane every vehicle counts 1 (lane holds 34 vehicles).

Comparisons: 3 (a 4-way maximum written out as a small knock-out).
"""
import numpy as np

LANES = ("N0", "N1", "N2", "E0", "E1", "E2", "S0", "S1", "S2", "W0", "W1", "W2")
UPSTREAM = ("N0u", "N1u", "E0u", "E1u", "S0u", "S1u", "W0u", "W1u")


def lane(x, name):
    """Stop-line lane load: halted vehicle = 1, moving vehicle = 0.5."""
    i = LANES.index(name)
    return 1.5 * (x[i] + x[12 + i])


def upstream(x, name):
    """Number of vehicles on an upstream lane."""
    return 34.1 * x[32 + UPSTREAM.index(name)]


# Preference added to each candidate's score, given the phase that is green now
# (candidate order: NS_through, NS_left, EW_through, EW_left; the entry for the current phase is the
# "extend" preference).
BONUS = {
    0: (-99.0, 0.5, 0.0, -0.45),  # after NS-through: never extend; NS-left favoured, EW-left slightly disfavoured
    1: (0.5, 0.55, 0.1, 0.1),     # after NS-left: extend only if nothing waits anywhere
    2: (0.0, -0.4, 1.1, -0.12),   # after EW-through: extend if the others score < ~1.1; NS-left disfavoured
    3: (0.0, -0.7, -0.55, 0.75),  # after EW-left: extend if the others score < ~0.75; NS-through favoured
}


def decide(x):
    x = np.asarray(x, dtype=float)
    current = int(np.argmax(x[40:44]))

    N0, N1, N2 = lane(x, "N0"), lane(x, "N1"), lane(x, "N2")
    E0, E1, E2 = lane(x, "E0"), lane(x, "E1"), lane(x, "E2")
    S0, S1, S2 = lane(x, "S0"), lane(x, "S1"), lane(x, "S2")
    W0, W1, W2 = lane(x, "W0"), lane(x, "W1"), lane(x, "W2")
    ns_through = N0 + S0 + S1
    ns_left = N2 + S2
    ew_through = E0 + E1 + W1
    ew_left = E2 + W2
    weak = 0.55 * (N1 + W0)                 # N1 and W0 count about half, equally for both through phases
    ew_up = upstream(x, "E0u") + upstream(x, "E1u") + upstream(x, "W0u") + upstream(x, "W1u")
    ns_up = upstream(x, "N0u") + upstream(x, "N1u") + upstream(x, "S0u") + upstream(x, "S1u")

    # own movement + spill-over credit for traffic elsewhere + upstream
    ns_through_score = N0 + S0 + 0.7 * S1 + weak + 0.2 * (ns_left + ew_left) + 0.03 * ns_up
    ns_left_score = 1.4 * N2 + 1.8 * S2 + 0.04 * ew_up
    ew_through_score = 0.65 * E0 + 1.05 * E1 + 0.8 * W1 + weak + 0.3 * (ns_through + ew_left) + 0.09 * ew_up
    ew_left_score = 1.3 * ew_left + 0.2 * (ew_through + ns_through + ns_left) + 0.09 * ew_up

    bonus = BONUS[current]
    s = [ns_through_score + bonus[0],
         ns_left_score + bonus[1],
         ew_through_score + bonus[2],
         ew_left_score + bonus[3]]

    a = 0 if s[0] >= s[1] else 1
    b = 2 if s[2] >= s[3] else 3
    return a if s[a] >= s[b] else b
