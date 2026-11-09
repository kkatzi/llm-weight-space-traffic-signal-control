"""Supervisor v1 (respect only): the network's highest-logit phase among the phases the rules allow now."""
import numpy as np


def decide(x, logits, state):
    lg = np.asarray(logits, dtype=float).reshape(-1)
    legal = np.asarray(state["legal"], dtype=bool).reshape(-1)
    if not legal.any():
        return int(np.argmax(lg))
    cand = np.flatnonzero(legal)
    return int(cand[np.argmax(lg[cand])])
