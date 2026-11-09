"""Supervisor on top of the trained PPO network, for deployment under the three operating rules.

    decide(x, logits, state) -> phase 0-3

What it does (details and reasoning in NOTES.md):

1. Respect the rules explicitly. It only ever returns a legal phase.
   - If exactly one phase is legal (minimum green running, or an overdue phase must be served), return it. The shield
     would do the same.
   - Otherwise start from the network's preferred LEGAL phase (highest logit among the legal ones), instead of the
     shield's "most halting vehicles" fallback.

2. Anticipate the pedestrian minimum green (the only intervention against a legal network choice).
   The network was trained with 12 s greens for every phase. Under the rules a newly started through phase (0 or 2)
   is held for 17 s (+3 s yellow = 20 s of junction time), while a left phase still takes 12 s (+3 s). Recorded runs
   show that through queues are gone by 12 s, so the 5 extra seconds mostly idle. Meanwhile the 3-car left-turn
   pockets keep filling for the whole 20 s and spill back into the shared straight+left upstream lane.
   So, when the network wants to START a through phase, the supervisor serves a left phase first, but only if
     - that left phase is legal, not the current green, and has at least MIN_LEFT_HALTING halting vehicles (it will
       use its 12 s and its pockets are close to spilling back), and
     - the network ranks it within THROUGH_MARGIN logits of the through phase (the network was nearly undecided), and
     - the through phase is not overdue.
   Holding the current green is never changed, and the network's choice is never changed when it picks a left phase,
   the other through phase, or when nothing is close.
"""
import numpy as np

THROUGH = (0, 2)
LEFT = (1, 3)
THROUGH_MARGIN = 4.0      # logits; how close a left phase must be ranked to the through phase the network wants
MIN_LEFT_HALTING = 4      # halting vehicles on the left phase's two pockets (6 fit), i.e. ~2 per pocket


def _vec(state, key, default, dtype):
    v = state.get(key, default) if isinstance(state, dict) else default
    return np.asarray(v, dtype=dtype).reshape(-1)


def decide(x, logits, state):
    lg = np.asarray(logits, dtype=float).reshape(-1)
    net = int(np.argmax(lg))
    legal = _vec(state, "legal", [True] * 4, bool)
    overdue = _vec(state, "overdue", [False] * 4, bool)
    halt = _vec(state, "phase_halting", [0.0] * 4, float)
    g = int(state.get("green", net)) if isinstance(state, dict) else net

    cand = [p for p in range(4) if legal[p]]
    if not cand:                       # should not happen; leave it to the shield
        return net
    if len(cand) == 1:                 # the rules leave no choice
        return int(cand[0])

    best = max(cand, key=lambda p: lg[p])          # the network's preferred legal phase
    if best in THROUGH and best != g and not overdue[best]:
        lefts = [a for a in LEFT if a != g and legal[a] and halt[a] >= MIN_LEFT_HALTING]
        if lefts:
            a = max(lefts, key=lambda p: lg[p])
            if lg[best] - lg[a] < THROUGH_MARGIN:
                return int(a)
    return int(best)
