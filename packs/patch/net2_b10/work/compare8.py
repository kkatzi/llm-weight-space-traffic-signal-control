import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "work"))
exec(open(os.path.join(HERE, "work", "compare.py")).read().split("for args in")[0])
from build import build_stayfix
E = dict(rho_left=2.0, rho_l1=1.0, rho_thr=3.0, mode_thr="avg", topk_left=48, topk_thr=40)
wE = build(**E); gE = tools.greedy(wE, X)
m3 = ok & (ph == 3)
for c in [0.5, 0.75, 1.0]:
    for sk in [16, 19]:
        w = build_stayfix(c, topk=sk, base=wE)
        g = tools.greedy(w, X)
        print(f"c={c} top{sk}: edit {tools.edit_size(w, w0)['changed']}, phase3 stays {(m3 & (g == 3)).sum()} (E {(m3 & (gE == 3)).sum()}), net eps {(m3 & (g == 3) & (ctrl == 0)).sum()}; "
              f"switch-target changes {(m3 & (gE != 3) & (g != gE)).sum()}; non-phase3 changes {(ok & (ph != 3) & (g != gE)).sum()}")
