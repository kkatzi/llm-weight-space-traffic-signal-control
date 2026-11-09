"""Build the L2W network by construction: W1 = W1_L2 @ M, b1 = b1_L2, rest copied.

M (46 x 54) maps an L2W observation x to a "virtual L2 observation" v = M @ x, i.e. what the L2 network
would have seen on a junction like L2. Nothing is fitted; every entry of M is set by the lane-role argument
in NOTES.md.

Usage:  python work/build.py [variant]     (default variant = "role")
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)
import tools  # noqa: E402

# L2 observation layout (46), in the order TASK.md gives
L2_SL = ["N0", "N1", "N2", "E0", "E1", "E2", "S0", "S1", "S2", "W0", "W1", "W2"]
L2_UP = ["N0u", "N1u", "E0u", "E1u", "S0u", "S1u", "W0u", "W1u"]
L2_NAMES = (["q_" + s for s in L2_SL] + ["d_" + s for s in L2_SL] + ["q_" + s for s in L2_UP]
            + ["d_" + s for s in L2_UP] + ["phase0", "phase1", "phase2", "phase3", "ok", "t_green"])
assert len(L2_NAMES) == 46

L2W_NAMES = tools.names()
assert len(L2W_NAMES) == 54

# N/S stop-line lanes are 19.2 m in L2W (capacity 2.56) but 22.4 m in L2 (capacity 2.987).
# Readings for 0,1,2,3+ vehicles: L2W 0, .391, .781, 1 ; L2 0, .335, .670, 1.
# Exact for 1-2 vehicles would be 2.56/2.987 = 0.857, exact for 3+ would be 1.0.
# Minimax over the three non-zero levels: .781 s - .670 = 1 - s  ->  s = 1.67/1.781 = 0.938.
CAP_L2 = 22.4 / 7.5
CAP_L2W_NS = 19.2 / 7.5
_l2 = [min(1.0, n / CAP_L2) for n in (1, 2, 3)]
_lw = [min(1.0, n / CAP_L2W_NS) for n in (1, 2, 3)]
NS_SCALE_MINIMAX = (_l2[1] + _l2[2]) / (_lw[1] + _lw[2])   # equalises the error at 2 and 3+ vehicles


def mapping(variant="role", ns_scale=None):
    """Return M (46 x 54): virtual L2 obs = M @ L2W obs."""
    if ns_scale is None:
        ns_scale = NS_SCALE_MINIMAX
    M = np.zeros((46, 54))
    li = {n: i for i, n in enumerate(L2_NAMES)}
    wi = {n: i for i, n in enumerate(L2W_NAMES)}

    def put(l2name, terms):
        for w_name, c in terms:
            M[li[l2name], wi[w_name]] += c

    for kind in ("q", "d"):
        # N and S: same lane structure, shorter stop-line lanes -> rescale the stop-line readings
        for ap in ("N", "S"):
            for k in range(3):
                put(f"{kind}_{ap}{k}", [(f"{kind}_{ap}{k}", ns_scale)])
            for k in range(2):
                put(f"{kind}_{ap}{k}u", [(f"{kind}_{ap}{k}u", 1.0)])
        # E and W: 4 stop-line lanes / 3 upstream lanes onto 3 / 2
        for ap in ("E", "W"):
            if variant == "role":
                # L2 lane 0 (right+straight, own upstream lane)  <- mean of L2W lanes 0 and 1 (own upstream lanes)
                # L2 lane 1 (straight, fed by the shared straight+left upstream lane) <- L2W lane 2
                # L2 lane 2 (left pocket) <- L2W lane 3
                put(f"{kind}_{ap}0", [(f"{kind}_{ap}0", 0.5), (f"{kind}_{ap}1", 0.5)])
                put(f"{kind}_{ap}1", [(f"{kind}_{ap}2", 1.0)])
                put(f"{kind}_{ap}2", [(f"{kind}_{ap}3", 1.0)])
                put(f"{kind}_{ap}0u", [(f"{kind}_{ap}0u", 0.5), (f"{kind}_{ap}1u", 0.5)])
                put(f"{kind}_{ap}1u", [(f"{kind}_{ap}2u", 1.0)])
            elif variant == "interp":
                # resample 3 through lanes onto 2 by lane position (each L2W lane gets total weight 2/3)
                put(f"{kind}_{ap}0", [(f"{kind}_{ap}0", 2 / 3), (f"{kind}_{ap}1", 1 / 3)])
                put(f"{kind}_{ap}1", [(f"{kind}_{ap}1", 1 / 3), (f"{kind}_{ap}2", 2 / 3)])
                put(f"{kind}_{ap}2", [(f"{kind}_{ap}3", 1.0)])
                put(f"{kind}_{ap}0u", [(f"{kind}_{ap}0u", 2 / 3), (f"{kind}_{ap}1u", 1 / 3)])
                put(f"{kind}_{ap}1u", [(f"{kind}_{ap}1u", 1 / 3), (f"{kind}_{ap}2u", 2 / 3)])
            elif variant == "drop":
                # ignore the extra straight lane (L2W lane 1 / 1u)
                put(f"{kind}_{ap}0", [(f"{kind}_{ap}0", 1.0)])
                put(f"{kind}_{ap}1", [(f"{kind}_{ap}2", 1.0)])
                put(f"{kind}_{ap}2", [(f"{kind}_{ap}3", 1.0)])
                put(f"{kind}_{ap}0u", [(f"{kind}_{ap}0u", 1.0)])
                put(f"{kind}_{ap}1u", [(f"{kind}_{ap}2u", 1.0)])
            elif variant == "sum":
                # total (not per-lane) through load: overstates EW by 1.5x
                put(f"{kind}_{ap}0", [(f"{kind}_{ap}0", 1.0), (f"{kind}_{ap}1", 0.5)])
                put(f"{kind}_{ap}1", [(f"{kind}_{ap}1", 0.5), (f"{kind}_{ap}2", 1.0)])
                put(f"{kind}_{ap}2", [(f"{kind}_{ap}3", 1.0)])
                put(f"{kind}_{ap}0u", [(f"{kind}_{ap}0u", 1.0), (f"{kind}_{ap}1u", 0.5)])
                put(f"{kind}_{ap}1u", [(f"{kind}_{ap}1u", 0.5), (f"{kind}_{ap}2u", 1.0)])
            else:
                raise ValueError(variant)
    for n in ("phase0", "phase1", "phase2", "phase3", "ok", "t_green"):
        put(n, [(n, 1.0)])
    return M


def build(variant="role", ns_scale=None):
    src = tools.load(os.path.join(PACK, "source_weights_L2.npz"))
    M = mapping(variant, ns_scale)
    w = {k: src[k].copy() for k in ("W2", "b2", "W3", "b3")}
    w["W1"] = src["W1"] @ M
    w["b1"] = src["b1"].copy()
    return w, M, src


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "role"
    w, M, src = build(variant)
    for k in ("W2", "b2", "W3", "b3"):
        assert np.array_equal(np.float32(w[k]), np.float32(src[k])), k
    out = os.path.join(HERE, "weights.npz")
    tools.save(w, out)
    # re-load and confirm the frozen layers are bit-identical to the source file
    z, s = np.load(out), np.load(os.path.join(PACK, "source_weights_L2.npz"))
    for k in ("W2", "b2", "W3", "b3"):
        assert z[k].dtype == s[k].dtype and np.array_equal(z[k], s[k]), k
    print("saved", out, "variant", variant, "ns_scale %.4f" % NS_SCALE_MINIMAX)
    print(tools.trainability(tools.load(out)))
