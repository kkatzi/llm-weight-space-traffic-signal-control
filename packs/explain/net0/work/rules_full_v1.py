"""Fuller readable rule set for the PPO signal controller (start_weights.npz).

Same core as rules_small.py: one demand score per phase from the stop-line vehicles, a handicap per
(current phase, candidate) pair, pick the best. Additions, all read from forward passes of the network on
light-traffic states:
  * the current phase may be kept, with its own handicap (only happens when its own lanes still hold
    halted vehicles, which is rare after 12 s of green);
  * a more detailed light-traffic hold on EW_through, including its dependence on how long EW_through
    has been green;
  * after NS_left in light traffic the network falls back to EW_through much more strongly.

Phases: 0 NS_through, 1 NS_left, 2 EW_through, 3 EW_left.
"""
import numpy as np

N0, N1, N2, E0, E1, E2, S0, S1, S2, W0, W1, W2 = range(12)

# Handicap of each candidate phase (score units = vehicles), relative to EW_through, given the current
# green. The entry for the current phase itself is the handicap for keeping it.
HANDICAP = {
    0: {0: -1.5, 1: -1.6, 2: 0.0, 3: -1.5},    # after NS_through
    1: {0: -1.8, 1: -1.0, 2: 0.0, 3: -0.6},    # after NS_left
    2: {0: -2.6, 1: -1.9, 2: 0.0, 3: -0.8},    # after EW_through
    3: {0: -2.1, 1: -0.8, 2: 0.0, 3: -0.6},    # after EW_left
}
# After NS_left with fewer than 4 vehicles at the stop lines, EW_through is a much stronger default.
HANDICAP_LIGHT_AFTER_NS_LEFT = {0: -3.4, 1: -1.0, 2: 0.0, 3: -1.5}


def light_traffic_from_ew_through(v, ns_through, ns_left, ew_through, green_s):
    """Decision while EW_through is green and the junction is nearly empty; None = use the scores.
    v: vehicles per stop-line lane (halted 1, moving 0.5)."""
    if v[E2] > 0.25:                        # any east left-turner: leave it to the scores (-> EW_left)
        return None
    if ns_left < 0.25:                      # no N/S left-turners
        if v[N0] > 0.75 and v[S0] > 0.75 and ns_through < 2.25 and green_s < 35:
            return 0                        # one halted car on both N0 and S0 calls NS_through early
        if v[W2] < 1.25 and ns_through < 2.25 + v[W2]:
            return 2                        # <= 2 N/S through cars (3 with a west left-turner): hold
        return None
    if ns_left < 1.25 and v[W2] < 0.25 and ns_through < 1.25:
        # one N/S left-turner (and at most one N/S through car)
        if ns_left < 0.75:
            return 2                        # still moving: hold
        if v[S2] > 0.25 and green_s >= 15:
            return 2                        # halted on S2: ignored once EW_through has run 15 s
        if v[N2] > 0.25 and green_s >= 55:
            return 2                        # halted on N2: ignored once EW_through has run ~1 min
        if ew_through < 0.25:
            return 1                        # otherwise served at once if no EW_through car is present
        return None
    if v[N2] < 0.25 and ns_left < 1.25 and v[W2] < 1.25 and ns_through < 0.25:
        return 2                            # one S2 left-turner plus one west left-turner: hold
    return None


def decide(x):
    x = np.asarray(x, dtype=float)
    cur = int(np.argmax(x[40:44]))
    green_s = 60.0 * x[45]
    # vehicles per stop-line lane: a halted vehicle counts 1, a moving one 0.5
    v = 1.5 * (x[0:12] + x[12:24])
    ns_through = v[N0] + v[N1] + v[S0] + v[S1]
    ns_left = v[N2] + v[S2]
    ew_through = v[E0] + v[E1] + v[W0] + v[W1]
    ew_left = v[E2] + v[W2]
    # vehicles on the four E/W upstream lanes (halted ones count double); N/S upstream lanes are ignored
    ew_upstream = 17.0 * (x[26] + x[27] + x[30] + x[31] + x[34] + x[35] + x[38] + x[39])

    if cur == 2:
        light = light_traffic_from_ew_through(v, ns_through, ns_left, ew_through, green_s)
        if light is not None:
            return light

    score = {
        0: ns_through - 0.6 * ew_left,
        1: 1.1 * ns_left,
        2: 0.6 * ew_through - 0.5 * ns_left + 0.1 * ew_upstream,
        3: ew_left,
    }
    total = ns_through + ns_left + ew_through + ew_left
    cand = HANDICAP_LIGHT_AFTER_NS_LEFT if (cur == 1 and total < 4) else HANDICAP[cur]
    return int(max(cand, key=lambda p: score[p] + cand[p]))
