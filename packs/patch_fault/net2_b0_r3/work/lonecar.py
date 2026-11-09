import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
nm = tools.names()
SERVE = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 0, 7: 0, 8: 1, 9: 2, 10: 2, 11: 3}

def lone_table(w, label, halting=True, ncar=1, fault=True):
    print(f"== {label}: lone {ncar} car(s) {'halting' if halting else 'moving'}; rows = lane, cols = (cur phase, t_green) -> chosen phase")
    tgs = [(0, 0.2), (1, 0.2), (1, 0.6), (1, 1.0)]
    hdr = "lane    " + " ".join(f"p{k}" + "".join(f"[{'ok' if o else 'no'}{tg:.1f}]" for o, tg in tgs) for k in range(4))
    print(hdr)
    bad = 0
    for lane in list(range(12)) + ["empty"]:
        row = []
        for k in range(4):
            cells = ""
            for o, tg in tgs:
                x = np.zeros(46)
                if lane != "empty":
                    v = min(1.0, ncar * 0.335) if ncar < 3 else 1.0
                    x[12 + lane] = v
                    if halting: x[lane] = v
                    if fault and lane == 11: x[11] = 0; x[23] = 0
                x[40 + k] = 1; x[44] = o; x[45] = tg
                a = int(tools.greedy(w, x)[0])
                cells += str(a)
            row.append(f"p{k}:{cells}")
        name = nm[lane] if lane != "empty" else "empty"
        print(f"{name:7s} " + "   ".join(row) + (f"   serve={SERVE[lane]}" if lane != "empty" else ""))

if __name__ == "__main__":
    w0 = tools.load("start_weights.npz")
    lone_table(w0, "start net", fault=False)
    lone_table(w0, "start net", halting=False, fault=False)
    lone_table(w0, "start net", ncar=2, fault=False)
