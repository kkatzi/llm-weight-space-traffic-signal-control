"""Supervisor v3: respect + order a soon-forced phase against the network's choice (see NOTES.md)."""
import numpy as np

THROUGH = (0, 2)
# nominal arrival rates per phase (veh/s), from TASK.md demand: through = straight + right, left = 30 %
LAM = np.array([2 * 450 * 0.70, 2 * 450 * 0.30, 2 * 750 * 0.70, 2 * 750 * 0.30]) / 3600.0


def _rule(state, key, default):
    r = state.get("rules", None)
    try:
        return float(r[key])
    except Exception:
        return default


def _slot(p):
    # seconds from the decision that starts p until p may end: 3 s yellow + 17 s (through) / 12 s (left)
    return 20.0 if p in THROUGH else 15.0


def decide(x, logits, state):
    lg = np.asarray(logits, dtype=float).reshape(-1)
    legal = np.asarray(state["legal"], dtype=bool).reshape(-1)
    if not legal.any():
        return int(np.argmax(lg))
    cand = np.flatnonzero(legal)
    q = int(cand[np.argmax(lg[cand])])
    g = int(state["green"])
    if len(cand) == 1 or q == g:
        return q
    halt = np.asarray(state["phase_halting"], dtype=float).reshape(-1)
    since = np.asarray(state["since_served"], dtype=float).reshape(-1)
    over = np.asarray(state["overdue"], dtype=bool).reshape(-1)
    if over.any():
        return q
    max_wait = 90.0
    # phases the rules will force right after q: halting now and reaching max wait before q may end
    soon = [p for p in range(4) if p not in (q, g) and legal[p] and halt[p] > 0 and since[p] + _slot(q) >= max_wait]
    if not soon:
        return q
    p = max(soon, key=lambda k: since[k])
    # both q and p will be served next; pick the order with less delay for the two queues
    cost_q_first = halt[p] * _slot(q) + LAM[p] * _slot(q) ** 2 / 2
    cost_p_first = halt[q] * _slot(p) + LAM[q] * _slot(p) ** 2 / 2
    return p if cost_p_first < cost_q_first else q
