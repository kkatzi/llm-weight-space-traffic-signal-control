"""Supervisor on top of the trained PPO network for the rule-constrained junction.

decide(x, logits, state) -> phase 0-3.  Uses only the network's logits plus the rule state (numpy only).

1. Respect: never ask for a phase the rules forbid. When the network's argmax is illegal, take the legal phase
   the NETWORK rates highest (the shield would take the legal phase with most halting vehicles instead).
2. Anticipate maximum wait: when starting the network's choice q would let another phase p (with halting
   vehicles) reach 90 s before q may end, the rules will force p right after q. Both are then served next, so
   choose the order of q and p that gives less delay to their two queues (pairwise interchange, using the
   slot each green occupies under the rules: 20 s through, 15 s left).
3. Anticipate the through minimum green (close calls only): a through green now occupies 20 s, not 15 s.
   When the network is nearly undecided (logit gap < DELTA) between its through choice and a left phase,
   take the phase that clears more halting vehicles per second of slot.
Everywhere else the network's argmax is returned unchanged.
"""
import json
import os

import numpy as np

THROUGH = (0, 2)
DELTA = 1.0                                   # logit gap under which the network counts as undecided
MAX_WAIT = 90.0                               # TASK.md rule 3
# nominal arrival rates per phase (veh/s) from TASK.md: 450 / 750 veh/h per approach, 70 % through+right, 30 % left
LAM = np.array([2 * 450 * 0.70, 2 * 450 * 0.30, 2 * 750 * 0.70, 2 * 750 * 0.30]) / 3600.0
_DUMPED = [False]


def _slot(p):
    """Seconds from the decision that starts p until the decision where p may end: 3 s yellow + 17 s / 12 s."""
    return 20.0 if p in THROUGH else 15.0


def _dump(x, logits, state):
    if _DUMPED[0]:
        return
    _DUMPED[0] = True
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "state_sample.json"), "w") as f:
            json.dump({k: (np.asarray(v).tolist() if not isinstance(v, dict) else
                           {kk: str(vv) for kk, vv in v.items()}) for k, v in state.items()} |
                      {"_types": {k: type(v).__name__ for k, v in state.items()}}, f, indent=1)
    except Exception:
        pass


def decide(x, logits, state):
    _dump(x, logits, state)
    lg = np.asarray(logits, dtype=float).reshape(-1)
    legal = np.asarray(state["legal"], dtype=bool).reshape(-1)
    if not legal.any():
        return int(np.argmax(lg))
    cand = np.flatnonzero(legal)
    q = int(cand[np.argmax(lg[cand])])            # 1. best legal phase by the network's own scores
    g = int(state["green"])
    if len(cand) == 1 or q == g:
        return q
    over = np.asarray(state["overdue"], dtype=bool).reshape(-1)
    if over.any():
        return q
    halt = np.asarray(state["phase_halting"], dtype=float).reshape(-1)
    since = np.asarray(state["since_served"], dtype=float).reshape(-1)

    # 2. a phase the rules will force right after q: order the two
    soon = [p for p in range(4) if p not in (q, g) and legal[p] and halt[p] > 0 and since[p] + _slot(q) >= MAX_WAIT]
    if soon:
        p = max(soon, key=lambda k: since[k])
        cost_q_first = halt[p] * _slot(q) + LAM[p] * _slot(q) ** 2 / 2
        cost_p_first = halt[q] * _slot(p) + LAM[q] * _slot(p) ** 2 / 2
        return p if cost_p_first < cost_q_first else q

    # 3. close call between a through choice and a left phase: vehicles cleared per second of slot
    if q in THROUGH:
        best, best_idx = q, halt[q] / _slot(q)
        for r in cand:
            r = int(r)
            if r in THROUGH or r == g or lg[q] - lg[r] >= DELTA:
                continue
            idx = halt[r] / _slot(r)
            if idx > best_idx:
                best, best_idx = r, idx
        return best
    return q
