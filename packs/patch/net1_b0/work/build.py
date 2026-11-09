"""Build the edited network by construction (no fitting).

Adds to the trained policy's logits a capped "exchange index" term
    offset_p = M * tanh(alpha/M * s(x)) projected on phase contrasts  -  alpha * G * lambda_p
where H_p = vehicles waiting for phase p (stop-line lane vehicles + halted upstream vehicles, shared
upstream lanes split 50/50) and lambda_p = nominal arrival rate of phase p (veh/s).
Three contrast channels (A: NS-thr vs EW-thr, B: NS-left vs EW-left, C: through vs left), each built from one
repurposed layer-1 unit (signal) and one repurposed layer-2 unit (relay).
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, tools

CS, CU = 22.4/7.5, 256/7.5
LAM = np.array([0.175, 0.075, 0.2917, 0.125])        # nominal veh/s per phase (630, 270, 1050, 450 veh/h)
N = tools.names(); IX = {n: i for i, n in enumerate(N)}

def H_coef():
    """Coefficient matrix C (4 x 46) with H = C @ x."""
    C = np.zeros((4, 46))
    for a in "NS":
        C[0, IX[f"d_{a}0"]] += CS; C[0, IX[f"d_{a}1"]] += CS; C[0, IX[f"q_{a}0u"]] += CU; C[0, IX[f"q_{a}1u"]] += 0.5*CU
        C[1, IX[f"d_{a}2"]] += CS; C[1, IX[f"q_{a}1u"]] += 0.5*CU
    for a in "EW":
        C[2, IX[f"d_{a}0"]] += CS; C[2, IX[f"d_{a}1"]] += CS; C[2, IX[f"q_{a}0u"]] += CU; C[2, IX[f"q_{a}1u"]] += 0.5*CU
        C[3, IX[f"d_{a}2"]] += CS; C[3, IX[f"q_{a}1u"]] += 0.5*CU
    return C

CH = {  # name: (signal as combination of H rows, logit direction)
    "A": (np.array([0.5, 0, -0.5, 0]), np.array([1, 0, -1, 0])),
    "B": (np.array([0, 0.5, 0, -0.5]), np.array([0, 1, 0, -1])),
    "C": (np.array([0.25, -0.25, 0.25, -0.25]), np.array([1, -1, 1, -1])),
}

def build(w0, alpha=0.75, M=2.5, G=10.0, eps=0.01, k_units=(31, 5, 61), j_units=(25, 35, 41),
          chans=("A", "B", "C"), X_mean=None):
    w = {k: v.copy() for k, v in w0.items()}
    C = H_coef()
    beta = alpha / (M * eps)
    # mean activations of the relay units before editing (for b3 compensation)
    h1, h2 = tools.hidden(w0, X_mean)
    for ch, k, j in zip(chans, k_units, j_units):
        comb, e = CH[ch]
        w["W1"][k, :] = eps * (comb @ C)
        w["b1"][k] = 0.0
        w["W2"][j, :] = 0.0
        w["W2"][j, k] = beta
        w["b2"][j] = 0.0
        w["b3"] = w["b3"] + w0["W3"][:, j] * h2[:, j].mean()
        w["W3"][:, j] = M * e
    w["b3"] = w["b3"] - alpha * G * LAM
    return w

def ideal_logits(w0, X, alpha=0.75, M=2.5, G=10.0, chans=("A", "B", "C")):
    lg = tools.forward(w0, X)
    Hm = X @ H_coef().T
    off = -alpha * G * LAM[None, :]
    for ch in chans:
        comb, e = CH[ch]
        s = Hm @ comb
        off = off + M * np.tanh(alpha / M * s)[:, None] * e[None, :]
    return lg + off

if __name__ == "__main__":
    w0 = tools.load("start_weights.npz")
    S = tools.states()
    X = S["obs"].astype(np.float64); ctl = S["controller"]
    ok = X[:, 44] > .5
    cfg = dict(alpha=float(sys.argv[1]) if len(sys.argv) > 1 else 0.75, M=float(sys.argv[2]) if len(sys.argv) > 2 else 2.5)
    w = build(w0, X_mean=X[ok], **cfg)
    g0 = tools.greedy(w0, X); g1 = tools.greedy(w, X); gi = ideal_logits(w0, X, **cfg).argmax(1)
    print("edit size", tools.edit_size(w, w0))
    for c in range(3):
        m = ok & (ctl == c)
        print(f"ctl{c}: changed vs orig {(g1[m]!=g0[m]).mean():.3f}  ideal changed {(gi[m]!=g0[m]).mean():.3f}  built!=ideal {(g1[m]!=gi[m]).mean():.3f}")
