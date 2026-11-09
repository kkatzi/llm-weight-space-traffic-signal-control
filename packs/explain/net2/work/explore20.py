from common import *
np.set_printoptions(linewidth=250, precision=2, suppress=True)
# synthetic grid: current phase p, set each group's lanes to a halting level (0..3 vehicles per lane, same on all lanes of the group)
GR = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
def make(p, lv, up=0.05):
    x = np.zeros(46); x[40+p] = 1; x[44] = 1; x[45] = 12/60; x[32:40] = up
    for g, l in lv.items():
        for i in GR[g]: x[i] = l/3; x[12+i] = l/3
    return x
for p in range(4):
    others = [j for j in range(4) if j != p]
    print(f"=== current phase {p}: decision grid. rows: level of group {others[0]}, cols: level of group {others[1]}, blocks: level of group {others[2]}")
    for l3 in range(4):
        print(f" group {others[2]} level {l3}")
        for l1 in range(4):
            row = []
            for l2 in range(4):
                x = make(p, {others[0]: l1, others[1]: l2, others[2]: l3})
                row.append(str(tools.greedy(w, x)[0]))
            print("   ", " ".join(row))
