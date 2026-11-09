"""Exhaustive margin check of the two detector units on a grid of 'edge' inputs: for every phase, every decision
green age, ok, and E2 in {empty, 1 moving car, 1 halting car, full}, with the rest of the traffic inputs at the
worst case for the unit's old pre-activation."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools, numpy as np
import build_full as B
w0 = tools.load("start_weights.npz")
X = tools.states()["obs"].astype(np.float64)
w = B.build(w0, X)
ages = [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 32, 37, 42, 47, 52, 55, 57, 60, 62]
worst = {}
for name, u, j, normal in [("A", B.U_A, B.J_A, 1.0), ("B", B.U_B, B.J_B, -1.0)]:
    r = w["W1"][u].copy(); b = w["b1"][u]
    bad_normal = 0; bad_cond = 0; n = 0
    for k in range(4):
        for g in ages:
            t = min(g / 60, 1.0); okv = 1.0 if g >= 10 else 0.0
            for e2 in [(0, 0), (0, 0.335), (0.335, 0.335), (1, 1)]:
                # traffic inputs other than E2: choose worst case per sign (both directions)
                rest = np.ones(40, bool); rest[[5, 17]] = False
                fixed = b + r[40 + k] + r[44] * okv + r[45] * t + r[5] * e2[0] + r[17] * e2[1]
                hi = fixed + np.clip(r[:40][rest], 0, None).sum()
                lo = fixed + np.clip(r[:40][rest], None, 0).sum()
                cond = (k == 2 and t >= B.THETA_A) if name == "A" else (k == 3 and t >= B.THETA_B and e2 == (0, 0))
                n += 1
                if cond:
                    # u should be at -normal: need sign(-normal) * pre >= 3 for both extremes
                    if min(-normal * hi, -normal * lo) < 3:
                        bad_cond += 1; print("     weak condition margin at phase", k, "age", g, "E2", e2, "pre range", round(lo, 2), round(hi, 2))
                else:
                    if min(normal * hi, normal * lo) < 3: bad_normal += 1
    print(f"pathway {name}: {n} edge cases, normal-state margin < 3: {bad_normal}, condition margin < 3: {bad_cond}")
