"""rules_small.py: compact readable translation of start_weights.npz (PPO actor, junction J0).

What the network does, in one sentence: every time the minimum green has elapsed it hands the green to the
phase with the largest weighted queue, where left-turn pockets count more per vehicle than through lanes, plus a
small preference that depends on the phase that is green now. It almost never extends the current green; it
does so only when the rest of the junction is (nearly) empty.

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
    0: (-99.0, 0.95, 0.0, 0.2),   # after NS-through: never extend; NS-left next is favoured
    1: (0.3, 0.9, 0.56, 0.76),    # after NS-left: extend only if nothing waits anywhere
    2: (0.0, -1.0, 1.4, -0.05),   # after EW-through: extend if the others hold < ~1.4 vehicles; NS-left disfavoured
    3: (0.0, -0.9, -0.6, 1.2),    # after EW-left: extend if the others hold < ~1.2 vehicles; NS-through favoured
}


def decide(x):
    x = np.asarray(x, dtype=float)
    current = int(np.argmax(x[40:44]))

    ns_through = lane(x, "N0") + lane(x, "S0") + lane(x, "S1")
    ns_left = lane(x, "N2") + lane(x, "S2")
    ew_through = lane(x, "E0") + lane(x, "E1") + lane(x, "W1")
    ew_left = lane(x, "E2") + lane(x, "W2")
    weak = 0.5 * (lane(x, "N1") + lane(x, "W0"))       # N1 and W0 count half, for both through phases
    ew_up = upstream(x, "E0u") + upstream(x, "E1u") + upstream(x, "W0u") + upstream(x, "W1u")

    bonus = BONUS[current]
    s = [ns_through + weak + bonus[0],
         1.6 * ns_left + bonus[1],
         ew_through + weak + 0.1 * ew_up + bonus[2],
         1.3 * ew_left + 0.1 * ew_up + bonus[3]]

    a = 0 if s[0] >= s[1] else 1
    b = 2 if s[2] >= s[3] else 3
    return a if s[a] >= s[b] else b
