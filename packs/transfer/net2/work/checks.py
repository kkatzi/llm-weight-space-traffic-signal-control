"""Checks of the transferred network (forward passes only; no simulator, no fitting)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)
sys.path.insert(0, HERE)
import tools  # noqa: E402
from build import build, embed, N46  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=160)

S = tools.states()
X = S["obs"].astype(np.float64)
L = tools.states("states_L2.npz")
Y = L["obs"].astype(np.float64)
N54 = tools.names()
ok = X[:, N54.index("ok")] > 0.5


def summary(tag, w):
    a = tools.greedy(w, X)
    ph = X[:, 48:52].argmax(1)
    sw = (a != ph) & ok
    print(f"[{tag}] act dist on L2W states {np.bincount(a, minlength=4) / len(a)} | "
          f"switch requests when ok: {sw.sum() / ok.sum():.3f} | "
          f"agree fixed {np.mean(a[S['controller'] == 0] == S['act'][S['controller'] == 0]):.3f} "
          f"actuated {np.mean(a[S['controller'] == 1] == S['act'][S['controller'] == 1]):.3f}")
    return a


if __name__ == "__main__":
    w = tools.load(os.path.join(HERE, "weights.npz"))
    src = tools.load(os.path.join(PACK, "source_weights_L2.npz"))

    # 1. copied layers
    for k in ("W2", "b2", "W3", "b3"):
        assert np.array_equal(w[k], src[k]), k
    print("1. W2 b2 W3 b3 identical to source")

    # 2. embedding check: L2 states rewritten in L2W layout -> same decisions as the L2 net on L2
    for s_ns, mode in [(0.93, "A"), (2.56 / 2.986666, "A")]:
        wv, _ = build(s_ns, mode)
        Xe = embed(Y, s_ns, mode)
        a_src = tools.greedy(src, Y)
        a_new = tools.greedy(wv, Xe)
        lg_diff = np.abs(tools.forward(wv, Xe) - tools.forward(src, Y)).max()
        print(f"2. embedded L2 states, s_ns={s_ns:.3f} mode {mode}: decision agreement {np.mean(a_src == a_new):.4f}, "
              f"agreement where ok {np.mean((a_src == a_new)[Y[:, 44] > 0.5]):.4f}, max |logit diff| {lg_diff:.3f}")

    # 3. behaviour on L2W recorded states
    a = summary("delivered", w)
    a_src_L2 = tools.greedy(src, Y)
    print("   L2 net on its own L2 states: act dist", np.bincount(a_src_L2, minlength=4) / len(a_src_L2),
          "switch requests when ok", np.mean((a_src_L2 != Y[:, 40:44].argmax(1))[Y[:, 44] > 0.5]).round(3))

    # 4. sensitivity to mapping choices (agreement with delivered net on L2W states)
    for s_ns, mode in [(2.56 / 2.986666, "A"), (1.0, "A"), (0.93, "B"), (0.93, "C")]:
        wv, _ = build(s_ns, mode)
        av = tools.greedy(wv, X)
        print(f"4. variant s_ns={s_ns:.3f} mode {mode}: agreement with delivered {np.mean(av == a):.4f} "
              f"(where ok {np.mean((av == a)[ok]):.4f})")

    # 5. trainability
    print("5. trainability delivered, L2W states:", tools.trainability(w))
    print("   trainability source, L2 states:    ", tools.trainability(src, Y))
