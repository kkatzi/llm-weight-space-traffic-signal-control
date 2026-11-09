import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, tools
from impute import fold, W0
import drain_table as D

S = tools.states(); X = S["obs"].astype(float); c = S["controller"]; ok = X[:, 44] > 0.5
P3 = 43
cands = {
    "mirror": ({5: 1.0}, {17: 1.0}, 0, 0),
    # half mirror + half phase-conditional mean (start-network E2 means: not-p3 q .407 d .457, p3 q .056 d .219)
    "half_phase": ({5: .5, P3: .5 * (.056 - .407)}, {17: .5, P3: .5 * (.219 - .457)}, .5 * .407, .5 * .457),
    "mirror_floor": ({5: 1.0, P3: -0.335}, {17: 1.0, P3: -0.335}, 0.335, 0.335),
}
a_ref = tools.greedy(W0, X)
def main():
    for name, (aq, ad, cq, cd) in cands.items():
        w = fold(aq, ad, cq, cd)
        g = tools.greedy(w, X)
        print(f"{name:13s} changed {tools.edit_size(w, W0)['changed']}  honoured share",
              np.round(np.bincount(g[ok], minlength=4) / ok.sum(), 3), " differs from faulty %.3f" % (g[ok] != a_ref[ok]).mean())
        D.run(w, name) if len(sys.argv) > 1 else print("  empty:", D.table(w, {}), "| W1ux3:", D.table(w, {"W1u": 3}),
                                                        "| N0x1:", D.table(w, {"N0": 1}), "| W0x1:", D.table(w, {"W0": 1}))


if __name__ == "__main__":
    main()
