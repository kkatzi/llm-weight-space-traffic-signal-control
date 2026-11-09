import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'work')
import tools, numpy as np, json
from build3 import build
n = tools.names()
cfg = json.loads(sys.argv[1]) if len(sys.argv)>1 else {"c3":1.0,"cN":0.5}
w,_ = build(**cfg)
lane_phase = {0:0,1:0,2:1,3:2,4:2,5:3,6:0,7:0,8:1,9:2,10:2,11:3}
def obs(phase, g, extra):
    x = np.zeros(46); x[40+phase]=1; x[44]=1.0 if g>=10 else 0.0; x[45]=min(g/60,1)
    for k,v in extra.items(): x[k]=v
    x[11]=0; x[23]=0   # failed detector
    return x
def first(w,p0,extra,target,steps=60):
    p=p0; g=2
    if p==target: return 0
    for s in range(steps):
        a=tools.forward(w,obs(p,g,extra))[0].argmax()
        if a!=p and g>=10: p=a; g=2
        else: g+=5
        if p==target: return (s+1)*5
    return None
bad=0
for ncar in [1,2,3]:
    for lane in range(12):
        extra={lane:ncar/2.987, lane+12:ncar/2.987}
        r=[first(w,p0,extra,lane_phase[lane]) for p0 in range(4)]
        if any(t is None for t in r): bad+=1
        print(f"{ncar} car(s) in {n[lane][2:]:3s} -> phase {lane_phase[lane]}: s until served from 0..3: {r}")
# pairs of lone cars in different lanes: each served eventually?
def visits(w,p0,extra,steps=120):
    p=p0; g=2; seen=set([p])
    for s in range(steps):
        a=tools.forward(w,obs(p,g,extra))[0].argmax()
        if a!=p and g>=10: p=a; g=2
        else: g+=5
        seen.add(int(p))
    return seen
fail=[]
for l1 in range(12):
    for l2 in range(l1+1,12):
        extra={l1:1/2.987,l1+12:1/2.987,l2:1/2.987,l2+12:1/2.987}
        need={lane_phase[l1],lane_phase[l2]}
        for p0 in range(4):
            if not need <= visits(w,p0,extra): fail.append((n[l1],n[l2],p0))
print("lanes never served (single):", bad, " pair cases with a never-served phase (static queue):", len(fail), fail[:12])
