"""Write work/weights.npz from the final construction (build5 defaults = the network evaluated in call 5)."""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'work')
import tools, numpy as np, hashlib
from build5 import build
w, w0 = build()                       # alpha=1, k=6, c3=1.0, cN=0.5, tg0=0.2
tools.save(w, "work/weights.npz")
wf = tools.load("work/weights.npz"); wv = tools.load("work/v5_k6.npz")
print("edit size:", tools.edit_size(wf, w0))
print("identical to evaluated v5_k6.npz:", all(np.array_equal(wf[k], wv[k]) for k in wf))
print("changed entries per tensor:", {k: int((wf[k] != w0[k]).sum()) for k in wf})
cols = np.where((wf['W1'] != w0['W1']).any(0))[0]; print("W1 columns changed:", [tools.names()[c] for c in cols])
print("trainability:", tools.trainability(wf))
