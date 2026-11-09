"""Rule-aware decision layer on top of the trained PPO network.

    decide(x, logits, state) -> int   # the phase, 0-3

Principle: the network stays in charge. The supervisor only
  1. respects the rules: it never asks for a phase the rules forbid. When the network's top phase is illegal it
     takes the legal phase the NETWORK ranks highest (the shield would take the legal phase with the most halting
     vehicles instead);
  2. anticipates the rules, but only where the network itself is close to indifferent (near-ties in its logits):
     a. max wait: a phase that will be forced at the next switch opportunity anyway is served now, if the network
        rates it almost as high as its top choice and serving it first is the cheaper order (Smith's rule on the
        halting vehicles and the time each green blocks the junction);
     b. pedestrian minimum green: a through green now blocks the junction for 20 s instead of the 15 s the
        network was trained with, so in a near-tie between a through phase and a left phase the left phase
        (15 s) is taken when it has more halting vehicles per second of the junction time it blocks.
Only numpy and math are used; the network is used only through the logits it is given.
"""
import math

import numpy as np

THROUGH = (0, 2)
LEFT = (1, 3)

# Rule parameters (TASK.md). The legal / overdue flags in `state` are authoritative; these numbers are only used to
# look one switch ahead.
MIN_GREEN = (15.0, 10.0, 15.0, 10.0)   # pedestrian minimum green per phase, s
MAX_WAIT = 90.0                        # a phase with halting vehicles must be served within this, s
STEP = 5.0                             # decision interval, s
YELLOW = 3.0                           # yellow after a switch, s (new green is STEP - YELLOW = 2 s old next decision)

# Near-tie margins, in logit units. At free switch decisions the network's top-2 logit gap has median 5.6; a gap
# below 0.5 occurs in ~5 % of them, below 1.1 in ~10 %. The margins are kept inside that "indifferent" band.
TIE_MAX_WAIT = 0.75
TIE_THROUGH = 0.5


def _slot(p):
    """Seconds from switching to phase p until the first decision at which p's green may end."""
    k = 1
    while (STEP - YELLOW) + STEP * (k - 1) < MIN_GREEN[p]:
        k += 1
    return STEP * k            # through: 20 s, left: 15 s


def _vec(state, key, default):
    try:
        v = state[key]
        return [float(v[p]) for p in range(4)]
    except Exception:
        return [float(default)] * 4


def decide(x, logits, state):
    lg = np.asarray(logits, dtype=np.float64).reshape(-1)[:4]
    net = int(np.argmax(lg))

    try:
        legal = [bool(state["legal"][p]) for p in range(4)]
    except Exception:
        return net
    if not any(legal):
        return net

    # 1. respect the rules: the network's best legal phase
    if legal[net]:
        best = net
    else:
        best = max((p for p in range(4) if legal[p]), key=lambda p: lg[p])
    if sum(legal) == 1:
        return best                                    # minimum green running, or a single overdue phase

    overdue = [v > 0.5 for v in _vec(state, "overdue", 0.0)]
    if any(overdue):
        return best                                    # several overdue phases legal: network's pick among them

    try:
        green = int(state["green"])
    except Exception:
        return best
    since = _vec(state, "since_served", 0.0)
    halt = _vec(state, "phase_halting", 0.0)
    r = best

    def horizon(p):
        """Seconds until the next decision at which the green may change again, if p is chosen now."""
        return STEP if p == green else _slot(p)

    # 2a. max wait: q would be overdue (and forced) at the first decision after r's green may end.
    cands = []
    for q in range(4):
        if q == r or q == green or not legal[q] or halt[q] <= 0.0:
            continue
        if since[q] + horizon(r) < MAX_WAIT:
            continue
        near_tie = lg[r] - lg[q] <= TIE_MAX_WAIT
        q_first_cheaper = halt[q] * horizon(r) > halt[r] * _slot(q)     # strict: on equality, later is better
        if near_tie and q_first_cheaper:
            cands.append(q)
    if cands:
        return max(cands, key=lambda p: lg[p])

    # 2b. pedestrian minimum green: through greens now cost 20 s, left greens 15 s.
    if r in THROUGH and r != green:
        lefts = [l for l in LEFT if l != green and legal[l] and halt[l] > 0.0]
        if lefts:
            l = max(lefts, key=lambda p: lg[p])
            r_not_at_risk = (halt[r] <= 0.0) or (since[r] + _slot(l) < MAX_WAIT)
            l_first_cheaper = halt[l] * _slot(r) > halt[r] * _slot(l)       # halting per second of green
            if lg[r] - lg[l] <= TIE_THROUGH and r_not_at_risk and l_first_cheaper:
                return l

    return r
