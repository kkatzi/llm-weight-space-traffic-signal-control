"""Build the starting network by hand (no fitting).  Run:  python work/build.py

Layer 1
  units 0-7   demand units   F[p,j] = tanh(S1 * (D_p - CENTERS[j]))       two per phase p (j = 0, 1)
              D_p = rule.score_matrix() @ x   (vehicle-equivalent demand of phase p, see rule.py)
  units 8-11  phase units    P_p = tanh(P_GAIN*phase_p - P_GAIN/2)  = +-0.76
  units 12-15 min-green units G_p = tanh(GA*phase_p - GA*ok - GB): ~ +0.9 if p is the current green and the
              minimum green has not elapsed (ok = 0), ~ -0.9..-1 otherwise
Layer 2
  units 0-7   logit units    L[p,j] = tanh(S2 * (F[p,0] + F[p,1] + BETAP*P_p - ZC) + OFFS[j])
  units 8-11  min-green units H_p = tanh(G_p)
Output
  logit_p = A/2 * (L[p,0] + L[p,1]) + K0 * H_p
All other units keep their random PPO initialisation; their output weights stay at the tiny initial values."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
import tools, rule

S1 = 0.10                 # layer-1 slope per vehicle-equivalent
CENTERS = (4.0, 14.0)     # two demand units per phase, centred at these demand levels (linearises the sum)
P_GAIN = 2.0              # phase passthrough
S2 = 0.8                  # layer-2 slope
ZC = 0.3                  # centre of layer-2 input (typical F0+F1)
OFFS = (-0.25, 0.25)      # two logit units per phase with slightly different offsets
A = 10.0                  # logit scale of the demand path
GA, GB = 3.0, 1.5         # min-green gate
K0 = 2.5                  # min-green hold logit weight (ok = 0 bonus = 1.44*K0, ok = 1 residual = 0.04*K0)


def g_slope(D):
    return sum(S1 * (1 - np.tanh(S1 * (D - c)) ** 2) for c in CENTERS)


def build(A_=None, K0_=None, beta=None):
    A_ = A if A_ is None else A_
    K0_ = K0 if K0_ is None else K0_
    beta = rule.BETA if beta is None else beta
    w = tools.load(os.path.join(ROOT, "start_weights.npz"))
    M, c0 = rule.score_matrix()
    F, P, G, L, H = {}, {}, {}, {}, {}
    k = 0
    for p in range(4):
        for j, c in enumerate(CENTERS):
            w["W1"][k] = S1 * M[p]; w["b1"][k] = S1 * (c0[p] - c); F[p, j] = k; k += 1
    for p in range(4):
        w["W1"][k] = 0.0; w["W1"][k, 40 + p] = P_GAIN; w["b1"][k] = -P_GAIN / 2; P[p] = k; k += 1
    for p in range(4):
        w["W1"][k] = 0.0; w["W1"][k, 40 + p] = GA; w["W1"][k, 44] = -GA; w["b1"][k] = -GB; G[p] = k; k += 1
    # hold bonus BETA vehicles (at a typical demand level D = 8), entered next to the demand
    betap = beta * g_slope(8.0) / (2 * np.tanh(P_GAIN / 2))
    m = 0
    for p in range(4):
        for j, off in enumerate(OFFS):
            w["W2"][m] = 0.0; w["W2"][m, F[p, 0]] = S2; w["W2"][m, F[p, 1]] = S2; w["W2"][m, P[p]] = S2 * betap
            w["b2"][m] = -S2 * ZC + off; L[p, j] = m; m += 1
    for p in range(4):
        w["W2"][m] = 0.0; w["W2"][m, G[p]] = 1.0; w["b2"][m] = 0.0; H[p] = m; m += 1
    for p in range(4):
        w["W3"][p, :m] = 0.0
        for j in range(2): w["W3"][p, L[p, j]] = A_ / 2
        w["W3"][p, H[p]] = K0_
        w["b3"][p] = 0.0
    hg = lambda ok: np.tanh(np.tanh(GA - GA * ok - GB)) - np.tanh(np.tanh(-GA * ok - GB))
    return w, dict(F=F, P=P, G=G, L=L, H=H, betap=betap, gap_ok0=K0_ * hg(0.0), gap_ok1=K0_ * hg(1.0))


if __name__ == "__main__":
    w, info = build()
    tools.save(w, os.path.join(HERE, "weights.npz"))
    print({k: round(float(info[k]), 4) for k in ("betap", "gap_ok0", "gap_ok1")})
