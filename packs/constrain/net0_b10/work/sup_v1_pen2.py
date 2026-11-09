"""Supervisor on top of the trained PPO network, for deployment under the operating rules.

decide(x, logits, state) -> phase 0-3

Idea (see NOTES.md):
  * Respect the rules explicitly: only ever return a legal phase; among legal phases use the network's own
    preference order (its logits), not the shield's "most halting vehicles" fallback.
  * Anticipate the pedestrian minimum green: the network was trained with 12 s through greens, but under the rules a
    through green lasts 17 s (+3 s yellow), while left greens still last 12 s. Recorded data show the through queue is
    usually gone by 12 s, so the extra 5 s mostly idles. When the network asks to SWITCH to a through phase, that
    choice now costs ~20 s of junction time instead of ~15 s, so its value per second is lower than the network
    believes. We subtract a fixed margin THROUGH_PENALTY from the logits of through phases that would be newly
    started, so a left phase that the network ranks almost as high is served first. Holding the current green is
    never penalised (its minimum green has already been paid), and an overdue phase is never penalised.
"""
import numpy as np

THROUGH = (0, 2)
THROUGH_PENALTY = 2.0     # logit margin; reasoning in NOTES.md


def decide(x, logits, state):
    lg = np.asarray(logits, dtype=float).reshape(-1)
    net = int(np.argmax(lg))
    legal = np.asarray(state.get("legal", [True] * 4), dtype=bool).reshape(-1)
    overdue = np.asarray(state.get("overdue", [False] * 4), dtype=bool).reshape(-1)
    g = int(state.get("green", net))

    cand = [p for p in range(4) if legal[p]]
    if not cand:                      # should not happen; let the shield handle it
        return net
    if len(cand) == 1:                # forced by the rules (minimum green, or a single overdue phase)
        return cand[0]

    adj = lg.copy()
    for p in THROUGH:
        if p != g and legal[p] and not overdue[p]:
            adj[p] -= THROUGH_PENALTY
    return int(max(cand, key=lambda p: adj[p]))
