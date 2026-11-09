"""Variant A: impute the dead W2 detector with the E2 detector (x11 := x5, x23 := x17).
Folding: W1[:,5] += a*W1[:,11], W1[:,17] += a*W1[:,23] (128 params)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
a = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
out = sys.argv[2] if len(sys.argv) > 2 else 'work/proxy_a1.npz'
w0 = tools.load('start_weights.npz')
w = {k: v.copy() for k, v in w0.items()}
w['W1'][:, 5] += a * w0['W1'][:, 11]
w['W1'][:, 17] += a * w0['W1'][:, 23]
tools.save(w, out)
print(out, tools.edit_size(tools.load(out), w0))
