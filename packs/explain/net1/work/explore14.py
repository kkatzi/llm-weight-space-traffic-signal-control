from harness import *
np.set_printoptions(precision=2, suppress=True, linewidth=200)
def setlane(x, l, nveh, halted=True):
    x[ix['d_'+l]] = nveh/3; x[ix['q_'+l]] = nveh/3 if halted else 0
def base(p):
    x = np.zeros(46); x[40+p]=1; x[44]=1; x[45]=0.2; return x
# per-lane response curves: logit(own) - mean(other candidates) as lane goes 0..3 vehicles (halted), from empty network
lanes = ['N0','N1','N2','E0','E1','E2','S0','S1','S2','W0','W1','W2']
for p in range(4):
    print(f"cur {p}: logits as one lane fills 0,1,2,3 halted vehicles")
    for l in lanes:
        rows = []
        for k in range(4):
            x = base(p); setlane(x,l,k); rows.append(tools.forward(w,x)[0])
        rows = np.array(rows)
        print(f"  {l}: " + "  ".join(f"[{' '.join(f'{v:5.1f}' for v in r)}]" for r in rows))
