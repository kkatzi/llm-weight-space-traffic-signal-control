"""Writes the delivered work/weights.npz (= the evaluated candidate v3_full.npz). See build.py for the construction."""
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "work"))
import tools
from build import build
w = build(c_t=0.6, t0=0.2, k3=0.25, k_sp=3, topk=60)
es = tools.edit_size(w, tools.load(os.path.join(HERE, "start_weights.npz"))); print(es); assert es["changed"] <= 371
tools.save(w, os.path.join(HERE, "work", "weights.npz"))
