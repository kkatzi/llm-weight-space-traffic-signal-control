from try9 import *
np.set_printoptions(precision=1, suppress=True)
B = np.array([[-99, 0.4, 0.0, -0.3],[0.45, 0.5, 0.0, 0.2],[0.0, -0.4, 1.1, -0.12],[0.0, -0.5, -0.75, 0.75]])
def synth(p, lane_level, up_level, served_empty=True):
    x = np.zeros(46); x[40+p]=1; x[44]=1; x[45]=0.2
    x[0:24] = lane_level
    x[24:32] = up_level*0.5; x[32:40] = up_level
    if served_empty:
        served = {0:['N0','N1','S0','S1'],1:['N2','S2'],2:['E0','E1','W0','W1'],3:['E2','W2']}[p]
        for l in served: x[ix['q_'+l]] = 0; x[ix['d_'+l]] = 0.33
    return x
for lvl, up in [(1.0, 0.0), (1.0, 0.2), (1.0, 0.4), (0.67, 0.1), (0.33, 0.05)]:
    print(f"all red lanes at {lvl}, upstream density {up}:")
    for p in range(4):
        x = synth(p, lvl, up)
        print(f"   cur {p}: net logits {tools.forward(w,x)[0]} -> {tools.greedy(w,x)[0]};  rule scores {scores9(x)+B[p]} -> {int(np.argmax(scores9(x)+B[p]))}")
