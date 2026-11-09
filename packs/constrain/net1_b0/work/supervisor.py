"""Decision layer on top of the trained PPO network, for deployment under the three operating rules.

    decide(x, logits, state) -> phase (0-3)

Principle: the network stays in charge. The supervisor only
  (a) returns a legal phase itself (instead of letting the shield substitute the "most halting" phase): when the
      network's choice is forbidden, it takes the legal phase the network ranks highest;
  (b) breaks near-ties in the network's ranking (top-2 logits within DELTA) in favour of the option the rules make
      cheaper:
        1. a phase that would otherwise become overdue before the next switch opportunity (it would be forced in
           next anyway, pre-empting whatever the network wants then);
        2. a left phase over a through phase (a through green is now held 15 s, i.e. until the 17 s decision, while
           the recorded data show its queue is always gone by 12 s; a left green still ends at 12 s).
Only numpy and math are used. Any unexpected input falls back to the network's own choice (the shield then still
enforces the rules).
"""
import math

import numpy as np

THROUGH = (0, 2)
LEFT = (1, 3)

# Near-tie margin in logits (top-1 minus candidate). 1.0 = the network gives the candidate at least ~e^-1 of the
# top choice's odds, i.e. it is not confident. Median top-1/top-2 gap at free decisions is ~4.5.
DELTA = 1.0

DEFAULTS = {"min_green_through": 15.0, "min_green_left": 10.0, "max_green": 45.0, "max_wait": 90.0}
DECISION_STEP = 5.0     # s between decisions
YELLOW = 3.0            # s of yellow; a new green is 2 s old at the next decision
FIRST_T_GREEN = DECISION_STEP - YELLOW


def _get(d, key, default=None):
    try:
        return d[key]
    except Exception:
        return default


def _rule(rules, names, default):
    """Read a rule parameter if the rules dict provides it under one of the expected names, else the default."""
    if not isinstance(rules, dict):
        return default
    for n in names:
        if n in rules:
            v = rules[n]
            try:
                return float(v)
            except Exception:
                pass
    return default


def _vec(v, n=4, default=0.0):
    """state entries may be lists, arrays or dicts keyed by phase; return a list of n floats."""
    out = []
    for p in range(n):
        try:
            out.append(float(v[p]))
        except Exception:
            out.append(default)
    return out


def _first_switch_t_green(min_green):
    """t_green at the first decision where a green of at least min_green may end (decisions at 2, 7, 12, ...).
    The junction itself ignores a switch before 10 s of green."""
    min_green = max(min_green, 10.0)
    k = max(0, math.ceil((min_green - FIRST_T_GREEN) / DECISION_STEP - 1e-9))
    return FIRST_T_GREEN + DECISION_STEP * k


def _horizon(c, current, mg_through, mg_left):
    """Seconds from now until the next decision at which the green chosen now (c) may end."""
    if c == current:
        return DECISION_STEP
    mg = mg_through if c in THROUGH else mg_left
    return YELLOW + _first_switch_t_green(mg)            # 3 + 17 = 20 (through), 3 + 12 = 15 (left)


def decide(x, logits, state):
    lg = np.asarray(logits, dtype=np.float64).reshape(-1)
    net_choice = int(np.argmax(lg))
    try:
        return _decide(lg, net_choice, state)
    except Exception:
        return net_choice


def _decide(lg, net_choice, state):
    legal = [bool(v) for v in _vec(_get(state, "legal"), default=1.0)]
    allowed = [p for p in range(4) if legal[p]]
    if not allowed:                     # should not happen; let the shield handle it
        return net_choice
    if len(allowed) == 1:               # the rules leave no choice (minimum green, or a single overdue phase)
        return allowed[0]

    current = int(_get(state, "green", -1))
    since = _vec(_get(state, "since_served"))
    halting = _vec(_get(state, "phase_halting"))
    rules = _get(state, "rules", {})
    max_wait = _rule(rules, ("max_wait", "max_wait_s", "maximum_wait", "wait_max", "max_red"), DEFAULTS["max_wait"])
    mg_through = _rule(rules, ("min_green_through", "min_green_thru", "ped_min_green_through"),
                       DEFAULTS["min_green_through"])
    mg_left = _rule(rules, ("min_green_left", "ped_min_green_left"), DEFAULTS["min_green_left"])

    # (a) the network's preferred phase among those the rules allow
    c = max(allowed, key=lambda p: lg[p])
    if c == current:                    # the network wants to keep the green and may: keep it
        return c

    hor_c = _horizon(c, current, mg_through, mg_left)

    def soon_overdue(p, horizon):
        return halting[p] > 0 and since[p] + horizon >= max_wait

    # (b) near-tie alternatives: other legal switch targets the network ranks within DELTA of its choice
    alts = [p for p in allowed if p != c and p != current and lg[c] - lg[p] <= DELTA]
    if not alts:
        return c

    # b1: serve a phase now that would otherwise be forced in at the next switch opportunity
    if not soon_overdue(c, hor_c):
        due = [p for p in alts if soon_overdue(p, hor_c)]
        if due:
            return max(due, key=lambda p: (since[p], lg[p]))

    # b2: prefer the shorter (left) commitment over a through green that will be held 15 s, unless delaying the
    # through phase by one left green would make it overdue
    if c in THROUGH:
        lefts = [p for p in alts if p in LEFT and halting[p] >= 1
                 and not soon_overdue(c, _horizon(p, current, mg_through, mg_left))]
        if lefts:
            return max(lefts, key=lambda p: lg[p])

    return c
