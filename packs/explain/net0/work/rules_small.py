"""Small readable rule set for the PPO signal controller (start_weights.npz).

The network picks the next green phase by comparing four "demand scores", one per phase, built from the
vehicles waiting at the stop line. EW_through (phase 2) is the reference; every other phase carries a
handicap that depends on which phase is green now. In practice it almost never keeps phases 0, 1 or 3
once the minimum green has elapsed (their lanes are empty by then), but it holds EW_through in light
traffic, even with a few side-street vehicles waiting.

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


def decide(x):
    cur = int(np.argmax(x[40:44]))
    # vehicles per stop-line lane: a halted vehicle counts 1, a moving one 0.5
    v = 1.5 * (np.asarray(x[0:12]) + np.asarray(x[12:24]))
    ns_through = v[N0] + v[N1] + v[S0] + v[S1]
    ns_left = v[N2] + v[S2]
    ew_through = v[E0] + v[E1] + v[W0] + v[W1]
    ew_left = v[E2] + v[W2]
    # vehicles on the four E/W upstream lanes (halted ones count double); N/S upstream lanes are ignored
    ew_upstream = 17.0 * (x[26] + x[27] + x[30] + x[31] + x[34] + x[35] + x[38] + x[39])

    if cur == 2:
        # Light traffic while EW_through is green: hold it (no east left-turner, no N/S left-turner,
        # at most one west left-turner and at most two N/S through vehicles, three if a west
        # left-turner waits).
        if v[E2] + ns_left < 0.25 and v[W2] < 1.25 and ns_through < 2.25 + v[W2]:
            return 2
        # A single halted north left-turner on an otherwise empty junction is served at once.
        if v[N2] > 0.75 and ns_left + ns_through + ew_left + ew_through < 1.25:
            return 1

    score = {
        0: ns_through - 0.6 * ew_left,
        1: 1.1 * ns_left,
        2: 0.6 * ew_through - 0.5 * ns_left + 0.1 * ew_upstream,
        3: ew_left,
    }
    total = ns_through + ns_left + ew_through + ew_left
    cand = HANDICAP_LIGHT_AFTER_NS_LEFT if (cur == 1 and total < 4) else HANDICAP[cur]
    return int(max(cand, key=lambda p: score[p] + cand[p]))
