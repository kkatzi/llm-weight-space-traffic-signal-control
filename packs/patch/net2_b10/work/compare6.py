import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "work"))
exec(open(os.path.join(HERE, "work", "compare.py")).read().split("for args in")[0])
from build import build_stayfix
G = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
d = X[:, 12:24]*cap
D = np.stack([d[:, G[p]].sum(1) for p in range(4)], 1)
r = np.arange(len(X))
m3 = ok & (ph == 3)
lg0 = tools.forward(w0, X)
for c in [0.5, 1.0, 1.5, 2.0, 3.0]:
    for tk in [64, 32]:
        w = build_stayfix(c, topk=tk)
        g = tools.greedy(w, X)
        stays = m3 & (g == 3); stays0 = m3 & (g0 == 3)
        # among phase-3 states where original switched, does the target change?
        sw0 = m3 & (g0 != 3)
        changed_target = (sw0 & (g != g0)).sum()
        # among removed stays: picked phase D vs others
        rem = stays0 & (g != 3)
        other_changes = (ok & (ph != 3) & (g != g0)).sum()
        print(f"c={c} topk={tk}: phase3 stays {stays.sum()} (orig {stays0.sum()}, net eps {(stays & (ctrl==0)).sum()} vs {(stays0 & (ctrl==0)).sum()}); "
              f"switch-target changed {changed_target}/{sw0.sum()}; non-phase3 changes {other_changes}; removed-stay new pick mean D {D[r, g][rem].mean():.2f}, max-other D {np.delete(D[rem], 3, 1).max(1).mean():.2f}")
# remaining stays at c=1: what do they look like
w = build_stayfix(1.0)
g = tools.greedy(w, X)
for i in np.where(m3 & (g == 3))[0][:10]:
    print("  remaining stay D", D[i].round(0), "tg", round(X[i, 45]*60))
