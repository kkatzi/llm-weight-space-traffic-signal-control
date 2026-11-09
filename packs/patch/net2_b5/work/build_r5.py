"""r5: drain rules E0-E3 only (no R1), same construction as build2.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
import build2
w0 = tools.load('start_weights.npz')
X = tools.states()['obs'].astype(float)
w = build2.build(w0, X, use=('E0', 'E1', 'E2', 'E3'))
print(tools.edit_size(w, w0))
tools.save(w, sys.argv[1] if len(sys.argv) > 1 else 'work/r5.npz')
