import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'work')
import tools, numpy as np
from build2 import build
n=tools.names()
def obs(phase, g, extra={}):
    x = np.zeros(46); x[40+phase]=1; x[44]=1.0 if g>=10 else 0.0; x[45]=min(g/60,1)
    for k,v in extra.items(): x[k]=v
    return x
w0=tools.load("start_weights.npz")
w2,_=build(0.75)
for name,w in [("start",w0),("v2",w2)]:
    print(name)
    for lane in [0,1,6,7, 2,8]:
        for ph in [2,3]:
            row=[]
            for g in [12,27,42,62]:
                lg=tools.forward(w,obs(ph,g,{lane:1/3,lane+12:1/3}))[0]
                row.append(np.round(lg,2).tolist())
            print(f"  car {n[lane]} phase {ph}: logits at g=12,27,42,62: {row}")
# two cars?
for name,w in [("start",w0),("v2",w2)]:
    for k in [1/3,2/3,1.0]:
        lg=tools.forward(w,obs(2,62,{0:k,12:k}))[0]
        lg2=tools.forward(w,obs(2,62,{0:k,12:k,6:k,18:k}))[0]
        print(name, "N0 cars",round(k*3), "phase2 g62 argmax", lg.argmax(), "  with S0 too:", lg2.argmax())
