import sys; sys.path.insert(0, '.')
import tools, numpy as np
n = tools.names()
w0 = tools.load("start_weights.npz")
w1 = tools.load(sys.argv[1] if len(sys.argv)>1 else "work/v1_proxy.npz")
def obs(phase, g, extra):
    x = np.zeros(46); x[40+phase]=1; x[44]=1.0 if g>=10 else 0.0; x[45]=min(g/60,1)
    for k,v in extra.items(): x[k]=v
    return x
serves = {0:[0,1,3,4,6,7,9,10], 1:[2,8], 2:[3,4,9,10,0,1,6,7], 3:[5,11]}
# which phase serves lane i (stop line 0..11): N0,N1 ->0 ; N2->1; E0,E1->2; E2->3; S0,S1->0; S2->1; W0,W1->2; W2->3
lane_phase = {0:0,1:0,2:1,3:2,4:2,5:3,6:0,7:0,8:1,9:2,10:2,11:3}
def run(w, p0, extra, steps=40):
    p=p0; g=2; seq=[]
    for s in range(steps):
        x=obs(p,g,extra)
        a=tools.forward(w,x)[0].argmax()
        if a!=p and g>=10: p=a; g=2
        else: g+=5
        seq.append(int(p))
    return seq
for name,w in [("start",w0),("cand",w1)]:
    print(name)
    for lane in range(12):
        extra={lane:1/3, lane+12:1/3}
        res=[]
        for p0 in range(4):
            seq=run(w,p0,extra)
            target=lane_phase[lane]
            t = next((i for i,p in enumerate(seq) if p==target), None)
            res.append(t)
        print(f"  car in {n[lane]} (needs phase {lane_phase[lane]}): steps until served from phase 0..3: {res}")
