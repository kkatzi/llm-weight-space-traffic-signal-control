"""Variant v3: through-start penalty 4.0, applied only when the left phase that would be served instead has at least
4 halting vehicles (so the 15 s it costs are well used, and its 3-car pockets are close to spilling back into the
shared straight+left upstream lane during a 20 s through service)."""
import numpy as np

THROUGH = (0, 2)
LEFT = (1, 3)
THROUGH_PENALTY = 4.0
MIN_LEFT_HALTING = 4


def decide(x, logits, state):
    lg = np.asarray(logits, dtype=float).reshape(-1)
    net = int(np.argmax(lg))
    legal = np.asarray(state.get("legal", [True] * 4), dtype=bool).reshape(-1)
    overdue = np.asarray(state.get("overdue", [False] * 4), dtype=bool).reshape(-1)
    halt = np.asarray(state.get("phase_halting", [0] * 4), dtype=float).reshape(-1)
    g = int(state.get("green", net))

    cand = [p for p in range(4) if legal[p]]
    if not cand:
        return net
    if len(cand) == 1:
        return cand[0]

    best = max(cand, key=lambda p: lg[p])          # the network's preferred legal phase
    if best in THROUGH and best != g and not overdue[best]:
        lefts = [a for a in LEFT if a != g and legal[a] and halt[a] >= MIN_LEFT_HALTING]
        if lefts:
            a = max(lefts, key=lambda p: lg[p])
            if lg[best] - lg[a] < THROUGH_PENALTY:
                return int(a)
    return int(best)
