"""Fuller readable rule set for the PPO signal controller (start_weights.npz).

Same core as rules_small.py: one demand score per phase from the stop-line vehicles, a handicap per
(current phase, candidate) pair, pick the best. Additions, all read from forward passes of the network on
light-traffic states:
  * the current phase may be kept, with its own handicap (only happens when its own lanes still hold
    halted vehicles, which is rare after 12 s of green);
  * a separate light-traffic model while EW_through is green (fewer than ~4 vehicles at the stop
    lines), read from the network's one-vehicle responses, including how long EW_through has been green;
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


def light_traffic_calls(v, green_s):
    """Nearly empty junction (fewer than ~4 vehicles at the stop lines) while EW_through is green.
    Each entry is how strongly that phase is called, relative to holding EW_through (0); the highest
    wins, so EW_through is held unless some call is positive. Coefficients are the network's own
    one-vehicle responses at an empty junction, grouped and rounded. v: vehicles per stop-line lane."""
    t = (min(green_s, 60.0) - 12.0) / 48.0     # 0 at the first possible switch (12 s), 1 after 60 s
    ew_through = v[E0] + v[E1] + v[W0] + v[W1]
    return {
        0: 1.1 * (v[N0] + v[S0]) + 0.6 * (v[N1] + v[S1]) + 0.6 * (v[N2] + v[S2])
           - 0.5 * v[W2] - 0.2 * v[E2] - 0.3 * ew_through - 2.2 - 0.15 * t,
        1: 1.9 * v[N2] + 1.3 * v[S2] + 0.2 * v[N0] - 0.2 * v[S0] - 0.5 * (v[N1] + v[S1])
           - 0.4 * ew_through - 1.3 - 0.25 * t,
        2: 0.0,
        3: 1.5 * v[E2] + 0.9 * v[W2] + 0.8 * v[N2] - 0.4 * (v[N1] + v[S1])
           - 0.6 * ew_through - 1.15 + 0.27 * t,
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
    # vehicles on the four E/W upstream lanes (halted ones count double); N/S upstream lanes are ignored
    ew_upstream = 17.0 * (x[26] + x[27] + x[30] + x[31] + x[34] + x[35] + x[38] + x[39])

    total = ns_through + ns_left + ew_through + ew_left
    if cur == 2 and total < 4.25:
        calls = light_traffic_calls(v, green_s)
        return int(max(calls, key=calls.get))

    score = {
        0: ns_through - 0.6 * ew_left,
        1: 1.1 * ns_left,
        2: 0.6 * ew_through - 0.5 * ns_left + 0.1 * ew_upstream,
        3: ew_left,
    }
    cand = HANDICAP_LIGHT_AFTER_NS_LEFT if (cur == 1 and total < 4) else HANDICAP[cur]
    return int(max(cand, key=lambda p: score[p] + cand[p]))
