"""Variant 3: overdue anticipation, no through penalty.

If the network's chosen new phase p would let another phase q become overdue before p's green may end
(since_served[q] + time to p's earliest end >= 90 s, q has halting vehicles), then q will be forced right after p
anyway and will break the network's own plan for the green after p. Serve q now instead (the one the network
ranks highest if several), and give control back to the network afterwards.
"""
import numpy as np

THROUGH = (0, 2)
MAX_WAIT = 90.0
# seconds from a switch decision to the first decision at which the new green may end
# (3 s yellow, decisions every 5 s: through greens end at 17 s -> 20 s; left greens at 12 s -> 15 s)
TO_END = {0: 20.0, 1: 15.0, 2: 20.0, 3: 15.0}


def must_follow(p, green, since, halting):
    """Phases that will be overdue when a green of p (started now) may first end."""
    d = 5.0 if p == green else TO_END[p]
    return [q for q in range(4) if q != p and q != green and halting[q] > 0 and since[q] + d >= MAX_WAIT]


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
    p = max(switch, key=lambda k: logits[k])
    if any(state["overdue"]):
        return int(p)
    since = [float(v) for v in state["since_served"]]
    halting = [float(v) for v in state["phase_halting"]]
    M = [q for q in must_follow(p, green, since, halting) if legal[q]]
    if M:
        return int(max(M, key=lambda k: logits[k]))
    return int(p)
