import sys; sys.path.insert(0, '.')
import tools, numpy as np
w0 = tools.load("start_weights.npz")
w1 = tools.load("work/v1_proxy.npz")
def obs(phase, tg, ok=1.0, extra=None):
    x = np.zeros(46); x[40+phase]=1; x[44]=ok; x[45]=tg
    if extra:
        for k,v in extra.items(): x[k]=v
    return x
for name,w in [("start",w0),("proxy",w1)]:
    print(name)
    for p in range(4):
        row=[]
        for tg in [2/60,7/60,12/60,22/60,32/60,47/60,1.0]:
            x=obs(p,tg, ok=1.0 if tg>=10/60 else 0.0)
            lg=tools.forward(w,x)[0]
            row.append(f"{lg.argmax()}({lg.max()-np.sort(lg)[-2]:.1f})")
        print("  phase",p," ".join(row))
# closed-loop on empty network: simulate decision timeline
def run(w, p0, steps=60, extra=None):
    p=p0; g=2; seq=[]
    for s in range(steps):
        x=obs(p, min(g/60,1), ok=1.0 if g>=10 else 0.0, extra=extra)
        a=tools.forward(w,x)[0].argmax()
        if a!=p and g>=10:
            p=a; g=2
        else:
            g+=5
        seq.append(p)
    return seq
for p0 in range(4):
    print("empty, start phase",p0, "proxy seq:", run(w1,p0,40))
# with one car in W2 (invisible), same. with one car in E2:
print("one car in E2 (q=d=1/3) proxy:", run(w1,0,40,{5:1/3,17:1/3}))
