"""What does a weights file choose when every working detector reads (almost) empty?"""
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
import tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w = tools.load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "work", "v1.npz"))
for extra in ["empty", "W1u small queue"]:
    print("==", extra)
    for p in range(4):
        row = []
        for tg in [0.05, 0.17, 0.25, 0.5, 0.75, 1.0]:
            x = np.zeros(46); x[40 + p] = 1; x[45] = tg; x[44] = 1.0 if tg * 60 >= 10 else 0.0
            if extra != "empty":
                x[31] = 2 / 34.13; x[39] = 2 / 34.13
            lg = tools.forward(w, x)[0]
            row.append(f"tg={tg:.2f}->{lg.argmax()} ({lg.max()-np.sort(lg)[-2]:.2f})")
        print("phase", p, " ".join(row))
