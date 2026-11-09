"""Write the final weights (v2: twin substitution + time term) to work/weights.npz and check them."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "work"))
import tools, build_v2

w0, w = build_v2.build()
tools.save(w, os.path.join(HERE, "work", "weights.npz"))
wl = tools.load(os.path.join(HERE, "work", "weights.npz"))
print("edit size (saved file):", tools.edit_size(wl, w0))
wv = tools.load(os.path.join(HERE, "work", "v2_twin_time.npz"))
print("identical to evaluated v2_twin_time.npz:", all(np.array_equal(wl[k], wv[k]) for k in tools.KEYS))
changed = {k: int((wl[k] != w0[k]).sum()) for k in tools.KEYS}
print("changed per tensor:", changed)
cols = sorted(set(np.where((wl["W1"] != w0["W1"]).any(0))[0]))
print("changed W1 columns:", [tools.names()[c] for c in cols])
print("trainability:", tools.trainability(wl))
