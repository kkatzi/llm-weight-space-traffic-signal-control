"""Hand-built repair of start_weights.npz for the failed W2 detector (x[11] = q_W2, x[23] = d_W2 always 0).

Every value is set by construction (no fitting). Four parts:

1. Mirror proxy for the dead detector: the W2 pocket is served by the same phase (EW_left) as E2 and has the same
   demand, so q_E2 is the stand-in for both q_W2 and d_W2:
       W1[:, 5] += W1[:, 11] + W1[:, 23]
   i.e. every first-layer unit now receives q_E2 as if it were also q_W2 and d_W2.

2. IDLE unit (h1 unit 4, repurposed): +1 when no vehicle is visible anywhere except vehicles halted on W1u
   (those can only be left-turners blocked by a full, invisible W2), -1 otherwise.
       pre = B0 - K * (sum of all lane inputs except q_W1u, d_W1u, and the dead x11/x23) - K * (d_W1u - q_W1u)
   Cars left in W2 at the end of an episode are invisible; without this rule the network parks in EW_through
   forever (2 of 6 recorded episodes ran to the 90-min cap).

3. BLOCK unit (h1 unit 28, repurposed): +1 when the current green is EW_through (phase 2) and at least one vehicle
   halts on W1u, -1 otherwise.  On the visible E/N/S sides this signature means the left pocket is full in
   98-100 % of recorded states, so W2 is full and its left-turners block the shared upstream lane.
       pre = -3 - C + C * phase2 + A * q_W1u

4. RELAY unit (h2 unit 37, repurposed): reads only the IDLE and BLOCK units and adds only to the EW_left logit.
   Baseline (both sources at -1): pre = -3 -> h = -0.995, and b3 is compensated so the baseline adds nothing.

Old contributions of the repurposed units are removed: W2[:,4] and W2[:,28] are zeroed (except the relay row), the
relay's old logit contribution is replaced by its mean (folded into b3).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import tools  # noqa: E402

# ---- constants (chosen by hand; see NOTES.md) ----
U_IDLE, U_BLOCK, U_RELAY = 4, 28, 37
K = 300.0        # idle: one vehicle anywhere (min. nonzero input 1/34.1 = 0.029) gives -8.8 -> unit fully off
B0 = 3.0         # idle bias: on-state pre >= 2.96 -> h >= 0.995
A = 200.0        # block: one halted vehicle on W1u (0.029) gives +5.9
C = A + 2.0      # block: outside phase 2 the unit can never turn on (q_W1u <= 1)
BB = -3.0        # block bias inside phase 2
G_IDLE = 3.0     # relay gain from IDLE
G_BLOCK = None   # relay gain from BLOCK, set from P_BLOCK below
L = 4.0          # relay -> EW_left logit weight
P_BLOCK = float(os.environ.get("P_BLOCK", "6.0"))  # EW_left logit push when BLOCK fires (alone); see diag_block_push.py

Q_W1U, D_W1U = 31, 39
DEAD = (11, 23)
LANES = list(range(40))
PH2 = 42


def build(p_block=P_BLOCK, verbose=True):
    w0 = tools.load("start_weights.npz")
    w = {k: v.copy() for k, v in w0.items()}
    S = tools.states()
    X = S["obs"].astype(np.float64)

    # mean activation of the relay unit before repurposing, on recorded states with the proxy applied
    Xp = X.copy(); Xp[:, 11] = X[:, 5]; Xp[:, 23] = X[:, 5]
    _, h2_old = tools.hidden(w0, Xp)
    relay_mean = h2_old[:, U_RELAY].mean()

    # 1. mirror proxy (all units; units 4 and 28 are overwritten below)
    w["W1"][:, 5] = w0["W1"][:, 5] + w0["W1"][:, 11] + w0["W1"][:, 23]

    # 2. IDLE unit
    for i in LANES:
        if i in DEAD:
            continue  # input always 0: leave untouched (no effect, saves edits)
        w["W1"][U_IDLE, i] = -K
    w["W1"][U_IDLE, Q_W1U] = +K   # halted W1u vehicles do not count as "busy" ...
    w["W1"][U_IDLE, D_W1U] = -K   # ... moving ones do (d - q >= 0)
    w["b1"][U_IDLE] = B0
    # W1[4, 40:46] (phase, ok, t_green) kept: |sum| <= 0.8, too small to matter against B0 and K

    # 3. BLOCK unit
    for i in range(46):
        if i in DEAD:
            continue
        w["W1"][U_BLOCK, i] = 0.0
    w["W1"][U_BLOCK, Q_W1U] = A
    w["W1"][U_BLOCK, PH2] = C
    w["b1"][U_BLOCK] = BB - C

    # relay gains: baseline pre = -3; idle alone -> 2*G_IDLE - 3; block alone -> 2*G_BLOCK - 3
    h_base = np.tanh(-3.0)
    # block push = L * (tanh(2*g_b - 3) - h_base)  ->  g_b
    t = p_block / L + h_base
    t = np.clip(t, -0.999, 0.999)
    g_block = (np.arctanh(t) + 3.0) / 2.0

    # columns of the repurposed h1 units: zero (removes their old effect), relay row only
    w["W2"][:, U_IDLE] = 0.0
    w["W2"][:, U_BLOCK] = 0.0
    # 4. RELAY unit row
    w["W2"][U_RELAY, :] = 0.0
    w["W2"][U_RELAY, U_IDLE] = G_IDLE
    w["W2"][U_RELAY, U_BLOCK] = g_block
    w["b2"][U_RELAY] = G_IDLE + g_block - 3.0
    # relay output: only EW_left; compensate its baseline and the removed old contribution in b3
    old_col = w0["W3"][:, U_RELAY].copy()
    w["W3"][:, U_RELAY] = np.array([0.0, 0.0, 0.0, L])
    w["b3"] = w0["b3"] + old_col * relay_mean - w["W3"][:, U_RELAY] * h_base

    if verbose:
        print("relay old mean %.4f, g_block %.4f, edit size %s" % (relay_mean, g_block, tools.edit_size(w, w0)))
    return w, w0


if __name__ == "__main__":
    w, w0 = build()
    es = tools.edit_size(w, w0)
    assert es["changed"] <= 371, es
    out = os.path.join(HERE, "weights.npz")
    tools.save(w, out)
    # re-load the saved float32 file and re-count (float32 rounding must not add edits)
    w_saved = tools.load(out)
    print("saved", out, tools.edit_size(w_saved, w0))
