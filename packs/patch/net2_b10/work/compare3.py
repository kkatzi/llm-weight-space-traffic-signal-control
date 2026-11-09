import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "work"))
exec(open(os.path.join(HERE, "work", "compare.py")).read().split("for args in")[0])
for args in [dict(rho_l1=0.5, rho_thr=0), dict(rho_l1=0.5, rho_thr=1, mode_thr="avg"), dict(rho_l1=0.5, rho_thr=2, mode_thr="avg"),
             dict(rho_l1=0.5, mode_l1="avg", rho_thr=1, mode_thr="avg"),
             dict(rho_l1=0.5, rho_thr=1, mode_thr="avg", topk_left=48, topk_thr=40),
             dict(rho_l1=0.5, rho_thr=1, mode_thr="avg", topk_left=56, topk_thr=32),
             dict(rho_l1=0.5, rho_thr=1, mode_thr="avg", topk_left=40, topk_thr=40)]:
    report(str(args), build(**args))
# energy captured by top-k
w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
from build import src_vec, K
for lane in ["N2", "S2", "E2", "W2"]:
    v = src_vec(w0, lane, "same") + 0.5*src_vec(w0, lane[0] + "1", "same")
    a = np.sort(np.abs(v))[::-1]
    print(lane, "L1 share captured by top 32/40/48/56:", [round(a[:k].sum()/a.sum(), 2) for k in (32, 40, 48, 56)])
