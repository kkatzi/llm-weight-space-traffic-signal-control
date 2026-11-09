"""Construct the edited network by hand (no fitting).

Edit: two 'empty left pocket' gates.
  gate EW: ok==1 and both EW left pockets (E2, W2) empty  -> logit of phase 3 (EW_left) -= P
  gate NS: ok==1 and both NS left pockets (N2, S2) empty  -> logit of phase 1 (NS_left) -= P
Each gate uses one repurposed layer-1 unit (detector) and one repurposed layer-2 unit (relay).
Lost functions of the repurposed units are replaced by their mean contribution (bias compensation).
"""
import numpy as np, sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools

IDX = {n: i for i, n in enumerate(tools.names())}
A = 20.0      # detector sharpness (pre-activation margin = 10)
G = 20.0      # detector -> relay weight
P = 40.0      # logit penalty when the gate is on


def build(k3, j3, k1, j1, w0=None, S=None):
    w0 = tools.load("start_weights.npz") if w0 is None else w0
    S = tools.states() if S is None else S
    X = S["obs"].astype(np.float64)
    net = S["controller"] == 0                    # statistics from the network's own episodes
    h1, h2 = tools.hidden(w0, X)
    mu1 = h1[net].mean(0); mu2 = h2[net].mean(0)
    w = {k: v.copy() for k, v in w0.items()}
    gates = [(k3, j3, "d_E2", "d_W2", 3), (k1, j1, "d_N2", "d_S2", 1)]
    ks = [g[0] for g in gates]; js = [g[1] for g in gates]
    # 1) bias compensation in layer 2 for the removed old outputs of the detector units
    for k in ks:
        for i in range(64):
            if i in js:
                continue
            w["b2"][i] += w0["W2"][i, k] * mu1[k]
    # 2) layer-2 relay units lose their old function: compensate logits with the mean contribution
    for j in js:
        w["b3"] += w0["W3"][:, j] * mu2[j]
    for k, j, la, lb, ph in gates:
        # detector row: z = A*(ok - 3*d_a - 3*d_b - 0.5); +10 iff ok=1 and both pockets empty
        w["W1"][k, :] = 0.0
        w["W1"][k, IDX["ok"]] = A
        w["W1"][k, IDX[la]] = -3 * A
        w["W1"][k, IDX[lb]] = -3 * A
        w["b1"][k] = -0.5 * A
        # detector column: only feeds its relay
        w["W2"][:, k] = 0.0
        # relay row: centre the rest of the pre-activation, then add the detector strongly
        w["W2"][j, k] = G
    for k, j, la, lb, ph in gates:
        # centre relay: rest = W2[j,:] h1 (detectors zero contribution except own) -> shift bias to median of rest
        h1n = np.tanh(X @ w["W1"].T + w["b1"])
        rest = h1n @ w["W2"][j].T - w["W2"][j, k] * h1n[:, k]
        w["b2"][j] = -np.median(rest)
        # relay -> logit: off (h=-1) contributes +P/2, on (h=+1) contributes -P/2; cancel the off state in b3
        w["W3"][:, j] = 0.0
        w["W3"][ph, j] = -P / 2
        w["b3"][ph] -= P / 2
    return w


if __name__ == "__main__":
    k3, j3, k1, j1 = [int(v) for v in sys.argv[1:5]]
    w = build(k3, j3, k1, j1)
    out = os.path.join(HERE, "work", "weights.npz")
    tools.save(w, out)
    print("saved", out, tools.edit_size(tools.load(out), tools.load("start_weights.npz")))
