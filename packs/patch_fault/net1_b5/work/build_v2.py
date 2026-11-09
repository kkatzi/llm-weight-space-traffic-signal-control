"""v2 = v1 (W2 inputs re-routed onto E2) + two hand-built trigger units that force phase 3 (EW_left).

Why: with the W2 detector dead, W2 left-turners are invisible. v1 lets W2 be served only when E2 shows demand, so
(a) at the end of an episode a few W2 vehicles can be stranded forever (the network holds a phase while it sees an
empty junction), and (b) in traffic a full W2 pocket spills back into upstream lane W1u and blocks the W through
traffic behind it while the network keeps EW_through green.

Units (all in 'car' units: q_W1u has capacity 256/7.5 = 34.13 cars, d_W1 has capacity 22.4/7.5 = 2.987 cars):
  h1[J_LONG]  long-green trigger: +1 iff the current green is not phase 3 and t_green >= 0.9 (> 54 s; at the 5-s
              decision grid it fires from 57 s on), else -1.     pre = 100*(t_green - 0.9) - 200*phase3
  h1[J_SPILL] W2-spillback trigger: +1 iff phase 2 (EW_through) is green, >= 1 car is halting in W1u and the W1
              stop-line lane is empty, else -1.  On the E side (all detectors) this pattern means E2 is full 96 % of
              the time.  pre = 8*Q - 280*Dw + 272*phase2 - 276   (Q = halting cars on W1u, Dw = cars on W1)
  h2[K_RELAY] OR of the two triggers: pre = 20*(h1_long + h1_spill + 1) + (its untouched old inputs, |.| <= 11.7)
              -> -1 normally, +1 when either trigger fires.
  W3[:, K_RELAY] = (0, 0, 0, S): a firing trigger adds 2*S = 10 to the phase-3 logit.
Compensation: both h1 trigger units sit at -1 in normal operation instead of their old (varying) values; b2 absorbs
the mean shift (b2 += W2[:, j] * (mean_old(h1_j) + 1)), and b3 absorbs the old relay unit's mean contribution, so
normal decisions equal mean-ablating these three units (0.2-0.3 % of decisions each, measured on recorded states).
"""
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
import tools

J_LONG, J_SPILL, K_RELAY = 0, 31, 25
CAP_UP, CAP_SL = 256 / 7.5, 22.4 / 7.5
G, S = 20.0, 5.0

w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
w1 = tools.load(os.path.join(HERE, "work", "v1.npz"))
w = {k: v.copy() for k, v in w1.items()}

# statistics of the old units on recorded (non-gridlock) states, v1 weights
St = tools.states(); X = St["obs"].astype(float)
m = (St["controller"] > 0) | (St["episode"] < 2)
h1, h2 = tools.hidden(w1, X[m])
c_long, c_spill, c_relay = h1[:, J_LONG].mean(), h1[:, J_SPILL].mean(), h2[:, K_RELAY].mean()

# --- h1 long-green trigger
w["W1"][J_LONG, :] = 0.0
w["W1"][J_LONG, 45] = 100.0           # t_green
w["W1"][J_LONG, 43] = -200.0          # phase3 one-hot
w["b1"][J_LONG] = -90.0               # 100 * 0.9
# --- h1 W2-spillback trigger
a, b, cc, e = 8.0, 280.0, 272.0, -276.0
w["W1"][J_SPILL, :] = 0.0
w["W1"][J_SPILL, 31] = a * CAP_UP     # q_W1u
w["W1"][J_SPILL, 22] = -b * CAP_SL    # d_W1
w["W1"][J_SPILL, 42] = cc             # phase2 one-hot
w["b1"][J_SPILL] = e
# --- b2 compensation for every h2 unit except the relay
w["b2"] = w1["b2"] + w1["W2"][:, J_LONG] * (c_long + 1.0) + w1["W2"][:, J_SPILL] * (c_spill + 1.0)
# --- h2 relay
w["W2"][K_RELAY, J_LONG] = G
w["W2"][K_RELAY, J_SPILL] = G
w["b2"][K_RELAY] = G
# --- output
w["b3"] = w1["b3"] + w1["W3"][:, K_RELAY] * c_relay + np.array([0, 0, 0, S])
w["W3"][:, K_RELAY] = np.array([0, 0, 0, S])

R = np.abs(np.delete(w["W2"][K_RELAY], [J_LONG, J_SPILL])).sum()
print("relay residual bound", round(R, 2), "(margin", round(G - R, 2), ")")
print("old means", round(c_long, 3), round(c_spill, 3), round(c_relay, 3))
print("edit size vs start:", tools.edit_size(w, w0))
tools.save(w, os.path.join(HERE, "work", "v2.npz"))
