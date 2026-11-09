"""Supervisor on top of the trained network, deployed under the operating rules.

decide(x, logits, state) -> phase 0-3

Logic
1. If the rules leave only one legal phase (min green running, or a single overdue phase that must come next),
   the outcome is fixed by the shield whatever we ask; we pass the network's own request through unchanged
   (the shield then applies the rule). No override by the supervisor.
2. If the network wants to keep the current green and that is legal, keep it.
3. Otherwise the network wants a new phase. Among the legal phases other than the current one, pick the best by
   the network's logits, after subtracting THROUGH_PENALTY from through phases (0, 2). Reason: the pedestrian rule
   holds a new through green for 17 s instead of the 12 s the network was trained with (its greens almost always
   end at the first chance), so starting a through phase now takes 20 s of junction time instead of 15 s, while a
   left phase still takes 15 s. With the queues the network keeps (about 2 vehicles per lane, nothing upstream)
   the extra 5 s of through green mostly serves nobody. The penalty only flips close calls between a through and
   a left phase (Smith's rule: order by waiting vehicles / service time). Through vs through and left vs left
   comparisons are unchanged.
"""
import numpy as np

THROUGH = (0, 2)
THROUGH_PENALTY = 2.0      # logits; ~1 vehicle-equivalent on the network's scale


def decide(x, logits, state):
    logits = np.asarray(logits, dtype=float)
    net = int(np.argmax(logits))
    legal = [bool(v) for v in state["legal"]]
    green = int(state["green"])
    cand = [p for p in range(4) if legal[p]]
    if len(cand) <= 1:
        return net                       # forced by the rules; the shield applies it
    if net == green and legal[green]:
        return net                       # the network keeps the current green
    switch = [p for p in cand if p != green]
    if not switch:
        return green
    score = {p: logits[p] - (THROUGH_PENALTY if p in THROUGH else 0.0) for p in switch}
    return int(max(switch, key=lambda p: score[p]))
