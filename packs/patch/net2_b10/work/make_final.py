"""Build the delivered network (= candidate E): spillback patch only."""
import sys, os, json
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "work"))
import numpy as np, tools
from build import build
w = build(rho_left=2.0, rho_l1=1.0, rho_thr=3.0, mode_thr="avg", topk_left=48, topk_thr=40)
w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
out = os.path.join(HERE, "work", "weights.npz")
tools.save(w, out)
wl = tools.load(out)
print("edit size of saved file:", tools.edit_size(wl, w0))
wE = tools.load(os.path.join(HERE, "work", "cand_E.npz"))
print("identical to evaluated cand_E.npz:", all(np.array_equal(wl[k], wE[k]) for k in wl))
with open(os.path.join(HERE, "work", "predictions.json"), "w") as f:
    json.dump({"low": 34.5, "nominal": 44.7, "high": 54.2}, f)
