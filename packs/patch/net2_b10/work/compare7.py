import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "work"))
exec(open(os.path.join(HERE, "work", "compare.py")).read().split("for args in")[0])
from build import build_stayfix
E = dict(rho_left=2.0, rho_l1=1.0, rho_thr=3.0, mode_thr="avg")
gE = report("E", build(**E, topk_left=48, topk_thr=40))
for tl, tt, sk in [(48, 32, 32), (44, 36, 32), (40, 32, 64), (40, 40, 32), (48, 40, 16)]:
    wsp = build(**E, topk_left=tl, topk_thr=tt)
    g1 = tools.greedy(wsp, X)
    w = build_stayfix(0.5, topk=sk, base=wsp)
    g = report(f"E l{tl} t{tt} + stay{sk}", w)
    print(f"    spill-only differs from E: {(ok & (g1 != gE)).sum()} honoured;  combined: stays {(ok & (g == ph)).sum()} (E {(ok & (gE == ph)).sum()})")
