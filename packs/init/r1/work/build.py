"""Build the starting network by hand (no fitting).

Rule implemented: pick the phase with the largest score
    S_p = sum over its stop-line lanes   [ WH * halting + WM * moving ]          (vehicles)
        + sum over its upstream lanes    share * [ WUH * halting + WUM * moving ] (vehicles)
        + K * [p is the current green]                                      (keep bonus / switching cost)
Counts are recovered from the observation: stop-line lanes hold 22.4/7.5 vehicles, upstream lanes 256/7.5.

Left pockets (lanes *2) are weighted x LP. Upstream lane 1 is shared by the through and the left phase.

Network: layer-1 units 0..15 are 4 units per phase, tanh(ALPHA_j*(S_p - MU_j)) (same ALPHA_j, MU_j for every
phase; steep near 0 and shallow tails so the logit keeps rising up to ~40 vehicles); layer-2 units 0..7 are 2 copies
per phase of tanh(BETA*sum_j h1_pj + c_k) (near-linear); logit_p = GAMMA * sum_k h2_pk.
Every phase goes through the same monotone function, so argmax(logits) == argmax(S_p) (up to the tiny random
read-out of the untouched units). All other units keep their random initial weights.
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools

names = tools.names(); ix = {n: i for i, n in enumerate(names)}
CAP_S, CAP_U = 22.4 / 7.5, 256 / 7.5

# ---- rule parameters (vehicle units) ----
# WH/WM: halting / moving vehicle on a stop-line lane; WUH/WUM: halting / moving vehicle on an upstream lane.
# Upstream lane 1 (straight + left) is shared: its MOVING vehicles are split 50/50 between the through and the left
# phase (SPLIT_MOV_THROUGH), its HALTING vehicles 35/65 (SPLIT_HALT_THROUGH), because in the recorded data a queue
# on upstream lane 1 is most often caused by a full left pocket blocking it.
# LP: weight multiplier for the left pockets (3 cars only; once full, further left-turners block upstream lane 1).
# K: keep bonus for the current green, just under one vehicle: a single waiting vehicle elsewhere is enough to leave
#    an empty green, while a green whose lanes still carry vehicles is kept until another phase has more.
P = dict(WH=1.0, WM=1.0, WUH=1.0, WUM=0.25, SPLIT_MOV_THROUGH=0.5, SPLIT_HALT_THROUGH=0.35, LP=1.25, K=0.75,
         # network shape: layer-1 gains/offsets (score units), layer-2 gain/offsets, read-out scale
         ALPHA=(0.3, 0.2, 0.12, 0.06), MU=(1.0, 6.0, 13.0, 26.0), BETA=0.2, C2=(-0.2, 0.2), GAMMA=8.0)

STOP = {0: ["N0", "N1", "S0", "S1"], 1: ["N2", "S2"], 2: ["E0", "E1", "W0", "W1"], 3: ["E2", "W2"]}
THROUGH_OF = {0: "NS", 2: "EW"}; LEFT_OF = {1: "NS", 3: "EW"}


def score_matrix(P):
    """Return (Wsc [4,46], bsc [4]) with S = Wsc @ x + bsc (bsc = 0)."""
    Wsc = np.zeros((4, 46))
    for p in range(4):
        for l in STOP[p]:
            # WH*h + WM*(n-h) = WM*n + (WH-WM)*h ; h = CAP*q, n = CAP*d ; left pockets x LP
            g = P.get("LP", 1.0) if l.endswith("2") else 1.0
            Wsc[p, ix["q_" + l]] += g * CAP_S * (P["WH"] - P["WM"])
            Wsc[p, ix["d_" + l]] += g * CAP_S * P["WM"]
        if p in THROUGH_OF:
            ups = [(a + "0u", 1.0, 1.0) for a in THROUGH_OF[p]] + \
                  [(a + "1u", P["SPLIT_HALT_THROUGH"], P["SPLIT_MOV_THROUGH"]) for a in THROUGH_OF[p]]
        else:
            ups = [(a + "1u", 1 - P["SPLIT_HALT_THROUGH"], 1 - P["SPLIT_MOV_THROUGH"]) for a in LEFT_OF[p]]
        for l, sh, sm in ups:
            # sh*WUH*h + sm*WUM*(n-h) = sm*WUM*n + (sh*WUH - sm*WUM)*h
            Wsc[p, ix["q_" + l]] += CAP_U * (sh * P["WUH"] - sm * P["WUM"])
            Wsc[p, ix["d_" + l]] += CAP_U * sm * P["WUM"]
        Wsc[p, ix[f"phase{p}"]] += P["K"]
    return Wsc, np.zeros(4)


def build(P, start="start_weights.npz"):
    w = tools.load(start)
    w = {k: v.copy() for k, v in w.items()}
    Wsc, bsc = score_matrix(P)
    nmu = len(P["MU"]); nc = len(P["C2"])
    alphas = np.broadcast_to(np.asarray(P["ALPHA"], dtype=float), (nmu,))
    l1 = {}  # (p, j) -> unit
    u = 0
    for p in range(4):
        for j, mu in enumerate(P["MU"]):
            w["W1"][u] = alphas[j] * Wsc[p]
            w["b1"][u] = alphas[j] * (bsc[p] - mu)
            l1[(p, j)] = u; u += 1
    n1 = u
    l2 = {}
    v = 0
    for p in range(4):
        for k, c in enumerate(P["C2"]):
            w["W2"][v] = 0.0
            for j in range(nmu):
                w["W2"][v, l1[(p, j)]] = P["BETA"]
            w["b2"][v] = c
            l2[(p, k)] = v; v += 1
    n2 = v
    # read-out: my layer-2 units -> their phase logit; other units keep their tiny random read-out
    w["W3"][:, :n2] = 0.0
    for p in range(4):
        for k in range(nc):
            w["W3"][p, l2[(p, k)]] = P["GAMMA"]
    w["b3"][:] = 0.0
    return w, dict(n1=n1, n2=n2)


def rule_scores(P, X):
    Wsc, bsc = score_matrix(P)
    return np.atleast_2d(X) @ Wsc.T + bsc


if __name__ == "__main__":
    w, info = build(P)
    S = tools.states(); X = S["obs"].astype(np.float64)
    a_net = tools.greedy(w, X)
    a_rule = rule_scores(P, X).argmax(1)
    print("net == rule argmax on recorded states:", (a_net == a_rule).mean())
    print("trainability:", tools.trainability(w))
    print("edit size:", tools.edit_size(w, tools.load("start_weights.npz")))
    out = os.path.join(HERE, "weights.npz")
    tools.save(w, out)
    with open(os.path.join(HERE, "params.json"), "w") as f:
        json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()}, f, indent=1)
    print("saved", out)
