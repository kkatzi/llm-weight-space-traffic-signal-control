"""Supervisor on top of the trained PPO network, for deployment under the three operating rules.

    decide(x, logits, state) -> int   (phase 0-3)

Design (see NOTES.md for the reasoning):
  1. The network's own choice (argmax of logits) is kept whenever it is legal, except for one narrow
     anticipation of the maximum-wait rule (step 3).
  2. If the network's choice is illegal:
       - exactly one phase legal (minimum green running, or a single overdue phase): the shield maps every request
         to that phase, so the outcome is fixed; the network's choice is returned unchanged (identical outcome,
         no recorded override);
       - several phases legal (max green reached, or several phases overdue at once): the legal phase the network
         ranks highest (logits), ties broken by halting vehicles.
  3. Max-wait anticipation: at a free switch decision, if some other phase p will be overdue by the time the
     network's pick c can end its minimum green (so p is forced right after c anyway), the order (c, p) is
     compared with (p, c) by Smith's rule (load / time occupied). The supervisor serves p first only when p is
     clearly more urgent: halting(p)/T(p) > 1.5 * load(c)/T(c), with at least 2 halting vehicles on p.
Uses only numpy and math; the network is used only through the logits passed in.
"""
import math

import numpy as np

# Rule parameters from TASK.md (overridden by state["rules"] when it carries recognisable numeric keys).
_DEFAULTS = {"min_green_through": 15.0, "min_green_left": 10.0, "max_green": 45.0, "max_wait": 90.0}
YELLOW = 3.0          # s of yellow before a new green
STEP = 5.0            # s between decisions
FIRST_AGE = 2.0       # age of a new green at the first decision after the switch
LANE_CAP = 22.4 / 7.5  # vehicles per stop-line lane (density 1.0)
KAPPA = 1.5           # Smith-rule margin before overriding the network's order
MIN_HALT_P = 2.0      # the phase served early must have at least this many halting vehicles

# Stop-line density indices (x[12:24]) per phase: N0 N1 N2 E0 E1 E2 S0 S1 S2 W0 W1 W2
_PHASE_DENS = {0: [12, 13, 18, 19], 1: [14, 20], 2: [15, 16, 21, 22], 3: [17, 23]}


def _rule(rules, keys, default):
    if isinstance(rules, dict):
        for k in keys:
            v = rules.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0:
                return float(v)
    return default


def _params(rules):
    return {
        "min_through": _rule(rules, ["min_green_through", "min_through", "ped_min_green_through"],
                             _DEFAULTS["min_green_through"]),
        "min_left": _rule(rules, ["min_green_left", "min_left", "ped_min_green_left"], _DEFAULTS["min_green_left"]),
        "max_wait": _rule(rules, ["max_wait", "max_wait_s", "maximum_wait"], _DEFAULTS["max_wait"]),
    }


def _end_age(phase, prm):
    """Green age at the first decision where `phase` may end (decisions at ages 2, 7, 12, 17, ...)."""
    mg = prm["min_through"] if phase % 2 == 0 else prm["min_left"]
    mg = max(mg, 10.0)  # the junction itself ignores switches before 10 s
    return FIRST_AGE + STEP * math.ceil((mg - FIRST_AGE) / STEP)


def _occupy(phase, prm):
    """Seconds from a switch decision to the first decision where the new green `phase` may end."""
    return YELLOW + _end_age(phase, prm)


def _load(x, phase, halting):
    """Vehicles on the phase's stop-line lanes (moving or halting), whole vehicles, never below the halting count."""
    n = sum(round(min(max(float(x[i]), 0.0), 1.0) * LANE_CAP) for i in _PHASE_DENS[phase])
    return float(max(n, halting))


def decide(x, logits, state):
    lg = np.asarray(logits, dtype=np.float64).reshape(-1)
    net = int(np.argmax(lg))
    try:
        legal = np.asarray(state["legal"], dtype=bool).reshape(-1)
    except Exception:
        return net
    if legal.size != 4 or not legal.any():
        return net
    halt = np.asarray(state.get("phase_halting", np.zeros(4)), dtype=np.float64).reshape(-1)
    if halt.size != 4:
        halt = np.zeros(4)

    # --- network's choice is illegal -------------------------------------------------------------
    if not legal[net]:
        if legal.sum() == 1:
            return net  # the shield maps any request to the single legal phase: outcome is the same
        cand = [p for p in range(4) if legal[p]]
        return int(max(cand, key=lambda p: (lg[p], halt[p])))

    # --- network's choice is legal: max-wait anticipation ----------------------------------------
    try:
        green = int(state["green"])
        since = np.asarray(state["since_served"], dtype=np.float64).reshape(-1)
        over = np.asarray(state.get("overdue", np.zeros(4)), dtype=bool).reshape(-1)
    except Exception:
        return net
    if since.size != 4 or net == green or over.any():
        return net
    prm = _params(state.get("rules"))
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    t_c = _occupy(net, prm)
    urg_c = _load(x, net, halt[net]) / t_c
    best, best_urg = None, -1.0
    for p in range(4):
        if p in (net, green) or not legal[p] or halt[p] < MIN_HALT_P:
            continue
        if since[p] + t_c < prm["max_wait"]:
            continue  # p will not be forced right after the network's pick
        urg_p = halt[p] / _occupy(p, prm)
        if urg_p > KAPPA * urg_c and urg_p > best_urg:
            best, best_urg = p, urg_p
    return net if best is None else int(best)
