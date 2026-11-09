# Notes
Built: start network with output biases b3[0] += 2 and b3[3] -= 2 (2 parameters changed; mk.py makes it).
Why: the net picks phase 3 (EW_left) in 37% of decisions and phase 0 in 15%; I nudged it slightly toward NS_through and away from EW_left.
Checks: 4 dev evals (baseline 44.8; +-1 shift 44.5 / reverse 44.8; +-2 shift 44.5, low 34.4 nominal 44.6 high 54.4).
Expected problems: the gain (~0.3 veh-h) is within seed noise; I predict essentially the baseline. A few episodes show occasional 1 leftover vehicle / extra switching.
