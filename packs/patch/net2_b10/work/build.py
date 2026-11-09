"""Construct the spillback patch by hand (no fitting).

Idea: the trained net sees at most 3 vehicles per stop-line lane (lanes are 22.4 m), and it almost ignores
the upstream-queue inputs q_X0u / q_X1u. When a queue spills back past the stop-line lane, those halted
upstream vehicles are invisible to it. We make it "count" them: each halted vehicle on an upstream lane is
presented to layer 1 as a virtual extra vehicle (queue + density) on the stop-line lane it is waiting for:
    q_X1u (feeds straight lane 1 AND the left pocket 2; blocked mostly by a full pocket) -> left pocket X2
    q_X0u (feeds lane 0 only)                                                          -> lane X0
This is done by adding scaled copies of existing W1 columns into the (almost always zero) q_up columns:
    W1[:, q_Xju] += rho * (cap_up / cap_stop) * (W1[:, q_src] + W1[:, d_src])
cap_up / cap_stop = (256/7.5) / (22.4/7.5) converts "one upstream vehicle" into "one stop-line vehicle".
Since q_up == 0 in ~96% of states, the network is unchanged there.
"""
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np, tools

NAMES = tools.names(); IX = {n: i for i, n in enumerate(NAMES)}
CAP, CAPU = 22.4/7.5, 256/7.5
K = CAPU / CAP
PAIR = {"N": "S", "S": "N", "E": "W", "W": "E"}


def src_vec(w0, lane, mode):
    """W1 direction of one extra vehicle on stop-line lane `lane` (e.g. 'S2').
    mode 'same': that lane only; 'avg': average of that lane and its mirror (same phase)."""
    v = w0["W1"][:, IX["q_" + lane]] + w0["W1"][:, IX["d_" + lane]]
    if mode == "avg":
        m = PAIR[lane[0]] + lane[1]
        v = 0.5 * (v + w0["W1"][:, IX["q_" + m]] + w0["W1"][:, IX["d_" + m]])
    return v


def build(rho_left=1.0, rho_thr=1.0, mode_left="same", mode_thr="same", topk_left=64, topk_thr=64, base=None,
          rho_l1=0.0, mode_l1="same"):
    w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
    w = {k: v.copy() for k, v in (base or w0).items()}
    for a in "NESW":
        for j, rho, mode, topk, src in [(1, rho_left, mode_left, topk_left, a + "2"),
                                        (0, rho_thr, mode_thr, topk_thr, a + "0")]:
            if rho == 0:
                continue
            add = rho * K * src_vec(w0, src, mode)
            if j == 1 and rho_l1:
                add = add + rho_l1 * K * src_vec(w0, a + "1", mode_l1)
            if topk < 64:
                keep = np.argsort(-np.abs(add))[:topk]
                mask = np.zeros(64, bool); mask[keep] = True
                add = np.where(mask, add, 0.0)
            w["W1"][:, IX[f"q_{a}{j}u"]] += add
    return w


if __name__ == "__main__":
    w = build()
    print(tools.edit_size(w, tools.load(os.path.join(HERE, "start_weights.npz"))))


def build_h2(rho=0.25, split=0.5, topk=64, base=None):
    """Anticipation patch: each MOVING vehicle on an upstream lane is shown to layer 1 as rho of a vehicle in
    the density input of the stop-line lane it will reach (up lane 0 -> lane 0; up lane 1 -> split between
    straight lane 1 and the left pocket 2). Adds scaled copies of W1 density columns into the d_up columns."""
    w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
    w = {k: v.copy() for k, v in (base or w0).items()}
    for a in "NESW":
        for j, vec in [(0, w0["W1"][:, IX[f"d_{a}0"]]),
                       (1, (1 - split) * w0["W1"][:, IX[f"d_{a}1"]] + split * w0["W1"][:, IX[f"d_{a}2"]])]:
            add = rho * K * vec
            if topk < 64:
                keep = np.argsort(-np.abs(add))[:topk]
                add = np.where(np.isin(np.arange(64), keep), add, 0.0)
            w["W1"][:, IX[f"d_{a}{j}u"]] += add
    return w


def build_stayfix(c=1.0, topk=64, base=None):
    """Phase-3 stay fix: while phase 3 (EW left) is green, layer 1 sees each EW left pocket with c vehicles
    fewer (queue + density) than it has: W1[:, phase3] += -c * (1/cap) * (W1[:,q_E2]+W1[:,d_E2]+W1[:,q_W2]+W1[:,d_W2]).
    The phase3 input is 0 in all other phases, so nothing else changes."""
    w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
    w = {k: v.copy() for k, v in (base or w0).items()}
    vec = sum(w0["W1"][:, IX[n]] for n in ["q_E2", "d_E2", "q_W2", "d_W2"]) / CAP
    add = -c * vec
    if topk < 64:
        keep = np.argsort(-np.abs(add))[:topk]
        add = np.where(np.isin(np.arange(64), keep), add, 0.0)
    w["W1"][:, IX["phase3"]] += add
    return w
