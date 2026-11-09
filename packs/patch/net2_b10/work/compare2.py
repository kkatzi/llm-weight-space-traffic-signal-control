import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "work"))
exec(open(os.path.join(HERE, "work", "compare.py")).read().split("for args in")[0])
for args in [dict(), dict(rho_thr=2.0), dict(rho_thr=2.0, mode_thr="avg"), dict(rho_l1=0.5), dict(rho_l1=1.0),
             dict(rho_thr=2.0, mode_thr="avg", rho_l1=0.5, mode_l1="avg"), dict(rho_thr=1.5, mode_thr="avg"),
             dict(rho_left=1.0, rho_thr=2.0, mode_thr="avg", topk_left=48, topk_thr=32)]:
    report(str(args), build(**args))
