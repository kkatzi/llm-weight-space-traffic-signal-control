"""Fuller readable rule set for the PPO signal controller (start_weights.npz).

Two regimes, both "score every phase, add a handicap that depends on the current green, take the best":
  * normal traffic (4 or more vehicles at the stop lines): the same demand scores and handicaps as
    rules_small.py (the handicaps are where the network's own decisions are tied, measured on the
    recorded states);
  * light traffic (fewer than ~4 vehicles): per-lane "calls" taken from the network's response to a
    single vehicle on an otherwise empty junction. This regime decides when EW_through is held with
    side-street vehicles waiting, which is how the network can stall (see EXPLANATION.md).
The current phase competes like the others, with its own handicap for being kept.

Phases: 0 NS_through, 1 NS_left, 2 EW_through, 3 EW_left.
"""
import numpy as np

N0, N1, N2, E0, E1, E2, S0, S1, S2, W0, W1, W2 = range(12)

# Normal traffic. Handicap of each candidate phase (score units = vehicles), relative to EW_through,
# given the current green. The entry for the current phase itself is the handicap for keeping it.
HANDICAP = {
    0: {0: -1.5, 1: -1.6, 2: 0.0, 3: -1.5},    # NS_through green
    1: {0: -1.8, 1: -1.0, 2: 0.0, 3: -0.6},    # NS_left green
    2: {0: -2.6, 1: -1.9, 2: 0.0, 3: -0.8},    # EW_through green
    3: {0: -2.1, 1: -0.8, 2: 0.0, 3: -0.6},    # EW_left green
}

# Light traffic. Call of each phase relative to EW_through at an empty junction, per current green.
LIGHT_BASE = {
    0: {0: -1.4, 1: -1.3, 2: 0.0, 3: -1.4},    # NS_through green
    1: {0: -2.4, 1: -1.4, 2: 0.0, 3: -1.4},    # NS_left green
    2: {0: -2.2, 1: -1.3, 2: 0.0, 3: -1.15},   # EW_through green
    3: {0: -2.2, 1: -0.6, 2: 0.0, 3: -0.6},    # EW_left green
}


def light_traffic_calls(cur, v, green_s):
    """How strongly each phase is called, relative to EW_through (0), on a nearly empty junction.
    The vehicle coefficients are the network's one-vehicle responses at an empty junction, grouped and
    rounded; they hardly depend on the current green. v: vehicles per stop-line lane."""
    t = (min(green_s, 60.0) - 12.0) / 48.0     # 0 at the first possible switch (12 s), 1 after 60 s
    ew_through = v[E0] + v[E1] + v[W0] + v[W1]
    base = LIGHT_BASE[cur]
    return {
        0: base[0] + 1.1 * (v[N0] + v[S0]) + 0.6 * (v[N1] + v[S1]) + 0.6 * (v[N2] + v[S2])
           - 0.5 * v[W2] - 0.2 * v[E2] - 0.3 * ew_through - 0.15 * t,
        1: base[1] + 1.9 * v[N2] + 1.3 * v[S2] + 0.2 * v[N0] - 0.2 * v[S0] - 0.5 * (v[N1] + v[S1])
           - 0.4 * ew_through - 0.25 * t,
        2: 0.0,
        3: base[3] + 1.5 * v[E2] + 0.85 * v[W2] + 0.8 * v[N2] - 0.4 * (v[N1] + v[S1])
           - 0.6 * ew_through + 0.27 * t,
    }


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

    if ns_through + ns_left + ew_through + ew_left < 4.25:
        calls = light_traffic_calls(cur, v, green_s)
        return int(max(calls, key=calls.get))

    # vehicles on the four E/W upstream lanes (halted ones count double); N/S upstream lanes are ignored
    ew_upstream = 17.0 * (x[26] + x[27] + x[30] + x[31] + x[34] + x[35] + x[38] + x[39])
    score = {
        0: ns_through - 0.6 * ew_left,
        1: 1.1 * ns_left,
        2: 0.6 * ew_through - 0.5 * ns_left + 0.1 * ew_upstream,
        3: ew_left,
    }
    cand = HANDICAP[cur]
    return int(max(cand, key=lambda p: score[p] + cand[p]))
