"""Supervisor on top of the trained network, deployed under the operating rules.

    decide(x, logits, state) -> phase 0-3

What it does
1. If the rules leave exactly one legal phase (a minimum green still running, or a single overdue phase that must
   come next), the outcome is fixed whatever is asked. The supervisor does not override: it passes the network's own
   request through and the shield applies the rule (the request is ignored by the junction in exactly the same way).
2. Otherwise (several legal phases) it returns the network's best LEGAL phase by its logits: the network's own
   choice when that is legal (the normal case), else the legal phase the network ranks highest. The shield would
   instead take the legal phase with the most halting vehicles; with several legal phases we keep the network's
   ranking (this only happens with max-green or two phases overdue at once, both rare).

What it deliberately does NOT do (tested in the simulator, see NOTES.md)
- No penalty on starting through phases for their longer 17 s minimum green (neutral at 2 logits, slightly worse
  at 4).
- No early service of phases that are about to become overdue (worse: +0.4 veh-h on dev, one high-demand episode
  broke down).
The network's phase sequence turned out to be hard to improve; the rules' cost is mostly structural (through greens
held 17 s instead of 12 s, forced service of the neglected NS-left phase), and the shield already handles both at
the latest legal moment.
"""
import numpy as np


def decide(x, logits, state):
    logits = np.asarray(logits, dtype=float)
    net = int(np.argmax(logits))
    legal = [bool(v) for v in state["legal"]]
    cand = [p for p in range(4) if legal[p]]
    if len(cand) <= 1:
        return net                       # outcome fixed by the rules; the shield applies it
    if legal[net]:
        return net                       # the network's own choice is allowed: keep it
    return int(max(cand, key=lambda p: logits[p]))   # best legal phase by the network's ranking
