"""Build the repaired network (build_full.build), save it to work/weights.npz and re-check the saved file."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools, numpy as np
import build_full as B
from chain import deadlock_report

w0 = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64)
w = B.build(w0, X)
out = os.path.join(HERE, "weights.npz")
tools.save(w, out)
ws = tools.load(out)
print("edit size (saved file):", tools.edit_size(ws, w0))
print("max |saved - float64 build|:", max(np.abs(ws[k] - w[k]).max() for k in tools.KEYS))
print("decision agreement saved vs float64 build on recorded states:", np.mean(tools.greedy(ws, X) == tools.greedy(w, X)))
changed = {k: int((ws[k] != w0[k]).sum()) for k in tools.KEYS}
print("changed per tensor:", changed)
print("trainability start:", tools.trainability(w0))
print("trainability saved:", tools.trainability(ws))
fails, emp = deadlock_report(ws, "saved weights", verbose=False)
print("lone-car deadlock chains (saved):", len(fails), "; empty-network chains visiting phase 3:", sum("3" in v[3:] for v in emp.values()), "/", len(emp))
