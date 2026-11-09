"""Build the L2W network (54 -> 64 -> 64 -> 4) from the L2 network (46 -> 64 -> 64 -> 4).

Construction (no fitting):  W1_new = W1_L2 @ M,  b1_new = b1_L2 + W1_L2 @ c,
where y = M x + c maps an L2W observation x (54) to an "equivalent L2 observation" y (46).
W2, b2, W3, b3 are copied unchanged from source_weights_L2.npz.

Usage:  python work/build.py [s_ns] [up_mode]
  s_ns    scale for the N/S stop-line lanes (L2W 19.2 m lanes vs L2 22.4 m lanes); default 0.93
  up_mode A: L2 E0u <- avg(E0u, E1u), L2 E1u <- E2u   (default)
          B: L2 E0u <- E0u,           L2 E1u <- avg(E1u, E2u)
          C: L2 E0u <- E0u,           L2 E1u <- E2u    (L2W E1u ignored)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)
import tools  # noqa: E402

N46 = ['q_N0', 'q_N1', 'q_N2', 'q_E0', 'q_E1', 'q_E2', 'q_S0', 'q_S1', 'q_S2', 'q_W0', 'q_W1', 'q_W2',
       'd_N0', 'd_N1', 'd_N2', 'd_E0', 'd_E1', 'd_E2', 'd_S0', 'd_S1', 'd_S2', 'd_W0', 'd_W1', 'd_W2',
       'q_N0u', 'q_N1u', 'q_E0u', 'q_E1u', 'q_S0u', 'q_S1u', 'q_W0u', 'q_W1u',
       'd_N0u', 'd_N1u', 'd_E0u', 'd_E1u', 'd_S0u', 'd_S1u', 'd_W0u', 'd_W1u',
       'phase0', 'phase1', 'phase2', 'phase3', 'ok', 't_green']
N54 = tools.names()

S_NS_DEFAULT = 0.93  # between 2.56/2.987 = 0.857 (count-exact for 1-2 cars) and 1.0 (exact for a full lane)


def mapping(s_ns=S_NS_DEFAULT, up_mode="A"):
    """Return M (46 x 54), c (46,) with y_L2 = M @ x_L2W + c."""
    M = np.zeros((46, 54))
    c = np.zeros(46)
    j = N54.index

    def put(l2name, terms):
        for name, coef in terms:
            M[N46.index(l2name), j(name)] += coef

    for k in ("q", "d"):
        for ap in ("N", "S"):                       # N/S: same 3 stop-line lanes, shorter (19.2 m)
            for ln in range(3):
                put(f"{k}_{ap}{ln}", [(f"{k}_{ap}{ln}", s_ns)])
        for ap in ("E", "W"):                       # E/W: 4 stop-line lanes -> 3
            put(f"{k}_{ap}0", [(f"{k}_{ap}0", 1.0)])                          # right+straight
            put(f"{k}_{ap}1", [(f"{k}_{ap}1", 0.5), (f"{k}_{ap}2", 0.5)])     # straight: mean of 2
            put(f"{k}_{ap}2", [(f"{k}_{ap}3", 1.0)])                          # left pocket
        for ap in ("N", "S"):                       # N/S upstream: identical
            for ln in range(2):
                put(f"{k}_{ap}{ln}u", [(f"{k}_{ap}{ln}u", 1.0)])
        for ap in ("E", "W"):                       # E/W upstream: 3 lanes -> 2
            if up_mode == "A":
                put(f"{k}_{ap}0u", [(f"{k}_{ap}0u", 0.5), (f"{k}_{ap}1u", 0.5)])
                put(f"{k}_{ap}1u", [(f"{k}_{ap}2u", 1.0)])
            elif up_mode == "B":
                put(f"{k}_{ap}0u", [(f"{k}_{ap}0u", 1.0)])
                put(f"{k}_{ap}1u", [(f"{k}_{ap}1u", 0.5), (f"{k}_{ap}2u", 0.5)])
            elif up_mode == "C":
                put(f"{k}_{ap}0u", [(f"{k}_{ap}0u", 1.0)])
                put(f"{k}_{ap}1u", [(f"{k}_{ap}2u", 1.0)])
            else:
                raise ValueError(up_mode)
    for n in ("phase0", "phase1", "phase2", "phase3", "ok", "t_green"):
        put(n, [(n, 1.0)])
    return M, c


def embed(Y, s_ns=S_NS_DEFAULT, up_mode="A"):
    """Inverse direction, for checks: an L2 observation written in L2W's layout (the extra straight lanes copy
    their neighbour, N/S stop-line values are converted car count -> L2W normalisation)."""
    Y = np.atleast_2d(Y)
    X = np.zeros((len(Y), 54))
    g = lambda n: Y[:, N46.index(n)]  # noqa: E731
    for k in ("q", "d"):
        for ap in ("N", "S"):
            for ln in range(3):
                cars = np.round(g(f"{k}_{ap}{ln}") * 22.4 / 7.5)
                X[:, N54.index(f"{k}_{ap}{ln}")] = np.minimum(cars / (19.2 / 7.5), 1.0)
        for ap in ("E", "W"):
            X[:, N54.index(f"{k}_{ap}0")] = g(f"{k}_{ap}0")
            X[:, N54.index(f"{k}_{ap}1")] = g(f"{k}_{ap}1")
            X[:, N54.index(f"{k}_{ap}2")] = g(f"{k}_{ap}1")
            X[:, N54.index(f"{k}_{ap}3")] = g(f"{k}_{ap}2")
        for ap in ("N", "S"):
            for ln in range(2):
                X[:, N54.index(f"{k}_{ap}{ln}u")] = g(f"{k}_{ap}{ln}u")
        for ap in ("E", "W"):
            if up_mode == "A":
                X[:, N54.index(f"{k}_{ap}0u")] = g(f"{k}_{ap}0u")
                X[:, N54.index(f"{k}_{ap}1u")] = g(f"{k}_{ap}0u")
                X[:, N54.index(f"{k}_{ap}2u")] = g(f"{k}_{ap}1u")
            elif up_mode == "B":
                X[:, N54.index(f"{k}_{ap}0u")] = g(f"{k}_{ap}0u")
                X[:, N54.index(f"{k}_{ap}1u")] = g(f"{k}_{ap}1u")
                X[:, N54.index(f"{k}_{ap}2u")] = g(f"{k}_{ap}1u")
            else:
                X[:, N54.index(f"{k}_{ap}0u")] = g(f"{k}_{ap}0u")
                X[:, N54.index(f"{k}_{ap}1u")] = g(f"{k}_{ap}0u")  # unused by the net in mode C
                X[:, N54.index(f"{k}_{ap}2u")] = g(f"{k}_{ap}1u")
    for n in ("phase0", "phase1", "phase2", "phase3", "ok", "t_green"):
        X[:, N54.index(n)] = g(n)
    return X


def build(s_ns=S_NS_DEFAULT, up_mode="A"):
    src = tools.load(os.path.join(PACK, "source_weights_L2.npz"))
    M, c = mapping(s_ns, up_mode)
    w = {"W1": src["W1"] @ M, "b1": src["b1"] + src["W1"] @ c,
         "W2": src["W2"], "b2": src["b2"], "W3": src["W3"], "b3": src["b3"]}
    return w, src


if __name__ == "__main__":
    s_ns = float(sys.argv[1]) if len(sys.argv) > 1 else S_NS_DEFAULT
    up_mode = sys.argv[2] if len(sys.argv) > 2 else "A"
    w, src = build(s_ns, up_mode)
    out = os.path.join(HERE, "weights.npz")
    tools.save(w, out)
    # exactness of the copied layers after the float32 round trip
    z_new, z_src = np.load(out), np.load(os.path.join(PACK, "source_weights_L2.npz"))
    for k in ("W2", "b2", "W3", "b3"):
        assert z_new[k].dtype == z_src[k].dtype and np.array_equal(z_new[k], z_src[k]), k
    print("saved", out, "| s_ns", s_ns, "| up_mode", up_mode, "| W2 b2 W3 b3 bit-identical to source")
