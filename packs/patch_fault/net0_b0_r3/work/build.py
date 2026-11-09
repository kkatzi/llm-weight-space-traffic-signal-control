"""Build the repaired network by construction.

Idea: the network was trained with q_W2 / d_W2 visible. Give it back a stand-in for them, built from inputs that
still work:
  (A) copy: q_W2 := q_E2 and d_W2 := d_E2 (the E and W left pockets have the same demand and are served by the same
      phase 3). Done exactly at layer 1: W1[:, q_E2] += W1[:, q_W2], W1[:, d_E2] += W1[:, d_W2].
  (B) spillback override: if at least one car is halted on W1u (the upstream lane feeding W1 and the W2 pocket)
      while phase 3 is not green, the W2 pocket is almost surely full (mirror lanes: 0.85-0.99 of the time).
      Then the network should see q_W2 = d_W2 = 1. This is a threshold, so it needs a tanh unit: layer-1 unit U
      (the least important one) is repurposed as the indicator
            h_U = tanh(A*q_W1u - B*phase3 - C)      (-1 off, +1 on)
      and its outgoing weights W2[:, U] inject v/2 (b2 gets +v/2), so the layer-2 pre-activation moves by v when
      the indicator is on and by 0 when off. v = mean, over recorded states where the indicator fires, of
      [layer-2 pre-activation with q_W2=d_W2=1] - [layer-2 pre-activation of the copy network]. That is an
      average of forward-pass activations, not a fit.
      The old contribution of unit U is replaced by its mean (folded into b2).
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import tools

nm = tools.names()
ix = {n: i for i, n in enumerate(nm)}
Q_E2, D_E2, Q_W2, D_W2 = ix["q_E2"], ix["d_E2"], ix["q_W2"], ix["d_W2"]
Q_W1U, PH3 = ix["q_W1u"], ix["phase3"]
ONE_CAR = 1 / (256 / 7.5)          # one halted vehicle on an upstream lane
U = 57                             # least important layer-1 unit (analyze5.py: 0.18% decisions change if frozen)
A, B, C = 170.0, 180.0, 2.5        # 0 cars -> tanh(-2.5)=-0.987, 1 car -> tanh(2.48)=+0.986; phase 3 -> off


def indicator(X):
    """The rule the repurposed unit implements (used for checks)."""
    return (X[:, Q_W1U] > 0.5 * ONE_CAR) & (X[:, PH3] < 0.5)


def a2_of(w, X):
    h1 = np.tanh(X @ w["W1"].T + w["b1"])
    return h1 @ w["W2"].T + w["b2"], h1


def build(v_subset="all"):
    w0 = tools.load("start_weights.npz")
    S = tools.states()
    X = S["obs"].astype(np.float64)

    # (A) copy E2 -> W2 at layer 1
    wc = {k: v.copy() for k, v in w0.items()}
    wc["W1"][:, Q_E2] += w0["W1"][:, Q_W2]
    wc["W1"][:, D_E2] += w0["W1"][:, D_W2]

    # (B) injection vector v
    fire = indicator(X)
    sel = fire if v_subset == "all" else fire & (X[:, Q_E2] < 0.99)
    Xt = X[sel].copy(); Xt[:, Q_W2] = 1.0; Xt[:, D_W2] = 1.0
    a2_target, _ = a2_of(w0, Xt)          # full-information network, pocket full
    a2_base, h1_base = a2_of(wc, X[sel])  # copy network as it would run
    v = (a2_target - a2_base).mean(0)

    w = {k: v_.copy() for k, v_ in wc.items()}
    _, h1_all = a2_of(wc, X)
    mean_old = h1_all[:, U].mean()
    # repurpose unit U
    w["W1"][U, :] = 0.0
    w["W1"][U, Q_W1U] = A
    w["W1"][U, PH3] = -B
    w["b1"][U] = -C
    w["b2"] = w["b2"] + w0["W2"][:, U] * mean_old + v / 2
    w["W2"][:, U] = v / 2
    return w, wc, v


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    w, wc, v = build(which)
    w0 = tools.load("start_weights.npz")
    print("edit size", tools.edit_size(w, w0))
    print("|v| =", np.linalg.norm(v).round(3))
    out = os.path.join(HERE, "weights.npz")
    tools.save(w, out)
    tools.save(wc, os.path.join(HERE, "weights_copy_only.npz"))
    print("saved", out)
