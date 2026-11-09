from common import *
w=W0
m=np.where((ctl==0)&ok)[0]
Xm=X[m]; base=tools.forward(w,Xm); gb=base.argmax(1)
print('finite diff: add one vehicle (halting) on stop lane / 3 halting on upstream; mean logit change (centered), and decision flips')
lanes=[n[2:] for n in N[0:12]]+[n[2:] for n in N[24:32]]
for k,ln in enumerate(lanes):
    Xp=Xm.copy()
    if k<12:
        qi,di=k,12+k; dv=1/cs
    else:
        qi,di=24+(k-12),32+(k-12); dv=3/cu
    Xp[:,qi]=np.clip(Xp[:,qi]+dv,0,1); Xp[:,di]=np.clip(Xp[:,di]+dv,0,1)
    lp=tools.forward(w,Xp); dl=(lp-base); dl=dl-dl.mean(1,keepdims=True)
    gp=lp.argmax(1)
    T=np.zeros(4,int)
    for p in gp[gp!=gb]: T[p]+=1
    print(f'{ln:5s}', dl.mean(0), 'flips to phase', T)
