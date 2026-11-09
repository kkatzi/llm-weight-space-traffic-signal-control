"""b3-only edit: add delta to the logits of the NS phases (0 and 1), correcting the EW-leaning training demand."""
import sys, numpy as np, tools
def build(delta, w0=None):
    w0 = tools.load("start_weights.npz") if w0 is None else w0
    w = {k: v.copy() for k, v in w0.items()}
    w["b3"] = w["b3"] + np.array([delta, delta, 0.0, 0.0]); return w
if __name__ == "__main__":
    d = float(sys.argv[1]); out = sys.argv[2]; w = build(d); tools.save(w, out)
    print(out, tools.edit_size(w, tools.load("start_weights.npz")))
