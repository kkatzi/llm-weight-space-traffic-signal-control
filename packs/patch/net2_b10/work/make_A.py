import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "work"))
import tools
from build import build
w = build(rho_left=1.0, rho_l1=0.5, rho_thr=1.0, mode_thr="avg", topk_left=48, topk_thr=40)
w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
print(tools.edit_size(w, w0))
tools.save(w, os.path.join(HERE, "work", "cand_A.npz"))
w2 = tools.load(os.path.join(HERE, "work", "cand_A.npz"))
print("after float32 save:", tools.edit_size(w2, w0))
