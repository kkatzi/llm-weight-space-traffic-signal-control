"""Build the edited network: three hidden units per layer are repurposed to compute a max-pressure readout.

Construction (by hand, no fitting):
  Phase pressures from stop-line lanes (queue = halting fraction, density = occupancy fraction):
    P0 (NS_through) = N0 + N1 + S0 + S1      P1 (NS_left) = N2 + S2
    P2 (EW_through) = E0 + E1 + W0 + W1      P3 (EW_left) = E2 + W2
  Stay bonus BETA: the score of phase p is P_p + BETA * [current green == p], which adds hysteresis.
  Layer 1, units I1..I3 (the three with the smallest outgoing weight mass in W2):
    h1[I_k] = tanh(ALPHA * (P_k - P3))          for k = 0,1,2   (W1 row rewritten, b1 = 0)
  Layer 2, units J1..J3 (the three with the smallest outgoing weight mass in W3):
    h2[J_k] = tanh(G * h1[I_k])                 (W2 row rewritten to a single entry, b2 = 0)
  Output: logit_k = K * h2[J_k] for k = 0,1,2 and logit_3 = 0 (+ old b3, unchanged).
  Because every step is monotone, argmax(logits) = argmax_p P_p (max-pressure), up to the old bias b3 and the
  small contribution of the untouched hidden units.

Edits: changed entries are counted with tools.edit_size.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)
import tools  # noqa: E402

ALPHA = 0.5     # layer-1 gain on pressure differences
G = 1.0         # layer-2 gain (single incoming weight)
K = 100.0       # output gain
DENS_W = 0.5    # weight of density relative to queue inside a pressure
BETA = 0.0      # stay bonus: the current green (one-hot phase, x[40:44]) gets +BETA pressure in the score
                # (dev sweep, eval calls: BETA 0 -> 46.5, 1 -> 47.1, 3 -> 57.3 veh-h; 0 is the best of these)

I_UNITS = [50, 42, 62]   # layer-1 units to repurpose (smallest ||W2[:, i]||_1)
J_UNITS = [1, 25, 2]     # layer-2 units to repurpose (smallest ||W3[:, j]||_1)

# stop-line lane indices (order from spec: N0 N1 N2 E0 E1 E2 S0 S1 S2 W0 W1 W2)
LANE = {"N0": 0, "N1": 1, "N2": 2, "E0": 3, "E1": 4, "E2": 5, "S0": 6, "S1": 7, "S2": 8, "W0": 9, "W1": 10, "W2": 11}
PHASE_LANES = {
    0: ["N0", "N1", "S0", "S1"],   # NS_through
    1: ["N2", "S2"],               # NS_left
    2: ["E0", "E1", "W0", "W1"],   # EW_through
    3: ["E2", "W2"],               # EW_left
}


def pressure_vector(p):
    """Weights on x (46,) for the pressure of phase p: queue + DENS_W * density on its stop-line lanes."""
    v = np.zeros(46)
    for name in PHASE_LANES[p]:
        i = LANE[name]
        v[i] += 1.0          # queue block (indices 0-11)
        v[12 + i] += DENS_W  # density block (indices 12-23)
    return v


def build(w):
    w = {k: np.array(v, dtype=np.float64, copy=True) for k, v in w.items()}
    # layer 1: row of unit I_k = ALPHA * (P_k - P_3)
    for k, i in enumerate(I_UNITS):
        row = pressure_vector(k) - pressure_vector(3)
        row[40 + k] += BETA      # current green = phase k earns the stay bonus (relative to phase 3)
        row[40 + 3] -= BETA      # ... and current green = phase 3 is penalised for the same reason
        w["W1"][i, :] = ALPHA * row
        w["b1"][i] = 0.0
    # layer 2: unit J_k reads only h1[I_k]
    for k, (j, i) in enumerate(zip(J_UNITS, I_UNITS)):
        w["W2"][j, :] = 0.0
        w["W2"][j, i] = G
        w["b2"][j] = 0.0
    # output: logit_k = K * h2[J_k] for k = 0,1,2; logit_3 gets nothing from these units
    for j in J_UNITS:
        w["W3"][:, j] = 0.0
    for k, j in enumerate(J_UNITS):
        w["W3"][k, j] = K
    return w


if __name__ == "__main__":
    ref = tools.load(os.path.join(PACK, "start_weights.npz"))
    new = build(ref)
    out = os.path.join(HERE, "weights.npz")
    tools.save(new, out)
    print("saved", out)
    print("edit size:", tools.edit_size(tools.load(out), ref))
