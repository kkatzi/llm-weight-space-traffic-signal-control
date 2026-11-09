"""Shared helpers for the work scripts."""
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
import tools
IDX = {n: i for i, n in enumerate(tools.names())}

def mirror(X):
    """Rotate an observation by 180 degrees (N<->S, E<->W); the W2 detector stays dead."""
    Y = X.copy()
    for a_, b_ in [("N", "S"), ("E", "W")]:
        for suf in ["0", "1", "2", "0u", "1u"]:
            for pre in ["q_", "d_"]:
                ia, ib = IDX[pre + a_ + suf], IDX[pre + b_ + suf]
                Y[:, ia], Y[:, ib] = X[:, ib].copy(), X[:, ia].copy()
    Y[:, 11] = 0; Y[:, 23] = 0
    return Y

def load_states():
    St = tools.states(); X = St["obs"].astype(float)
    return St, X
