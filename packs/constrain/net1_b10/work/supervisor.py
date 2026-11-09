"""Supervisor on top of the trained PPO network for the rule-constrained junction (final version).

decide(x, logits, state) -> phase 0-3.  Uses only the network's logits and the rule state; numpy + math only.

1. Respect: never ask for a phase the rules forbid. If the network's argmax is illegal, return the legal phase
   the NETWORK scores highest (the shield would take the legal phase with the most halting vehicles instead).
2. Anticipate the maximum wait: if starting the network's choice q would let another phase p (with halting
   vehicles) reach max_wait before q may end, the rules will force p right after q. Both are then served next,
   so choose the order of q and p that gives less delay to their two queues (pairwise interchange using the
   slot each green now occupies: 3 s yellow + earliest end, i.e. 20 s through, 15 s left).
3. Anticipate the through minimum green, in close calls only: a through green now occupies 20 s instead of
   15 s. When the network is nearly undecided (logit gap < DELTA) between its through choice and a legal left
   phase, take the phase that clears more halting vehicles per second of slot.
Everywhere else the network's own argmax is returned unchanged.
"""
import math

import numpy as np

THROUGH = (0, 2)
DELTA = 1.0                 # logit gap under which the network counts as undecided
YELLOW = 3.0                # s of yellow before a new green
FIRST_DECISION = 2.0        # a new green is 2 s old at the first decision; decisions every 5 s
STEP = 5.0
DEFAULT_MIN_GREEN = (15.0, 10.0, 15.0, 10.0)
DEFAULT_MAX_WAIT = 90.0
# nominal arrival rates per phase (veh/s) from TASK.md: 450 / 750 veh/h per approach, 70 % straight+right, 30 % left
LAM = np.array([2 * 450 * 0.70, 2 * 450 * 0.30, 2 * 750 * 0.70, 2 * 750 * 0.30]) / 3600.0


def _as_floats(v):
    if isinstance(v, str):
        v = v.strip("[]() ").split(",")
    return [float(t) for t in (v if isinstance(v, (list, tuple, np.ndarray)) else [v])]


def _rules(state):
    min_green, max_wait = list(DEFAULT_MIN_GREEN), DEFAULT_MAX_WAIT
    r = state.get("rules") or {}
    try:
        mg = _as_floats(r["min_green"])
        if len(mg) == 4:
            min_green = mg
    except Exception:
        pass
    try:
        max_wait = _as_floats(r["max_wait"])[0]
    except Exception:
        pass
    return min_green, max_wait


def _slot(p, min_green):
    """Seconds from the decision that starts p until the first decision at which p may end."""
    k = max(0, math.ceil((min_green[p] - FIRST_DECISION) / STEP))
    return YELLOW + FIRST_DECISION + STEP * k          # 20 s through, 15 s left under the TASK rules


def decide(x, logits, state):
    lg = np.asarray(logits, dtype=float).reshape(-1)
    legal = np.asarray(state["legal"], dtype=bool).reshape(-1)
    if not legal.any():
        return int(np.argmax(lg))
    cand = np.flatnonzero(legal)
    q = int(cand[np.argmax(lg[cand])])            # 1. best legal phase by the network's own scores
    g = int(state["green"])
    if len(cand) == 1 or q == g:
        return q
    if np.asarray(state["overdue"], dtype=bool).any():
        return q
    halt = np.asarray(state["phase_halting"], dtype=float).reshape(-1)
    since = np.asarray(state["since_served"], dtype=float).reshape(-1)
    min_green, max_wait = _rules(state)
    slot = [_slot(p, min_green) for p in range(4)]

    # 2. a phase the rules will force right after q: serve the two in the cheaper order
    soon = [p for p in range(4) if p not in (q, g) and legal[p] and halt[p] > 0 and since[p] + slot[q] >= max_wait]
    if soon:
        p = max(soon, key=lambda k: since[k])
        cost_q_first = halt[p] * slot[q] + LAM[p] * slot[q] ** 2 / 2
        cost_p_first = halt[q] * slot[p] + LAM[q] * slot[p] ** 2 / 2
        return p if cost_p_first < cost_q_first else q

    # 3. close call between a through choice and a left phase: halting vehicles cleared per second of slot
    if q in THROUGH:
        best, best_idx = q, halt[q] / slot[q]
        for r in cand:
            r = int(r)
            if r in THROUGH or r == g or lg[q] - lg[r] >= DELTA:
                continue
            idx = halt[r] / slot[r]
            if idx > best_idx:
                best, best_idx = r, idx
        return best
    return q
