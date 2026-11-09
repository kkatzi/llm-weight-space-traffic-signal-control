import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "work"))
exec(open(os.path.join(HERE, "work", "compare.py")).read().split("for args in")[0])
gA = report("A", build(rho_left=1.0, rho_l1=0.5, rho_thr=1.0, mode_thr="avg", topk_left=48, topk_thr=40))
for args in [dict(rho_left=2.0, rho_l1=1.0, rho_thr=2.0, mode_thr="avg", topk_left=48, topk_thr=40),
             dict(rho_left=1.5, rho_l1=0.75, rho_thr=1.5, mode_thr="avg", topk_left=48, topk_thr=40),
             dict(rho_left=2.0, rho_l1=0.5, rho_thr=2.0, mode_thr="avg", topk_left=48, topk_thr=40),
             dict(rho_left=3.0, rho_l1=1.5, rho_thr=3.0, mode_thr="avg", topk_left=48, topk_thr=40)]:
    g = report(str(args), build(**args))
    print("    differs from A in honoured:", (ok & (g != gA)).sum(), " in net eps:", (ok & (g != gA) & (ctrl == 0)).sum())
