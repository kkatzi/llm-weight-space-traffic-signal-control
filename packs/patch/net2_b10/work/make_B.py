import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "work"))
exec(open(os.path.join(HERE, "work", "compare.py")).read().split("for args in")[0])
from build import build_h2
for tk in [64, 40, 32]:
    w = build_h2(0.25, topk=tk)
    g = tools.greedy(w, X); ch = ok & (g != g0)
    print("topk", tk, tools.edit_size(w, w0)["changed"], "honoured changed", round(ch.mean(), 3), "net eps", round((ch & (ctrl == 0)).sum() / (ok & (ctrl == 0)).sum(), 3), "stays", (ok & (g == ph)).sum())
w = build_h2(0.25, topk=40)
tools.save(w, os.path.join(HERE, "work", "cand_B.npz"))
print(tools.edit_size(tools.load(os.path.join(HERE, "work", "cand_B.npz")), w0))
