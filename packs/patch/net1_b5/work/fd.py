from common import *
from mirror import perm, mirror
def fd_table(w, Xm, verbose=True):
    base=tools.forward(w,Xm); gb=base.argmax(1)
    out={}
    for i in range(40):
        dv = 1/cs if i<24 else 3/cu
        Xp=Xm.copy(); Xp[:,i]=np.clip(Xp[:,i]+dv,0,1)
        if i<12: Xp[:,12+i]=np.clip(Xp[:,12+i]+dv,0,1)   # halting vehicle also counts in density
        if 24<=i<32: Xp[:,8+i]=np.clip(Xp[:,8+i]+dv,0,1)
        lp=tools.forward(w,Xp); dl=lp-base; dl=dl-dl.mean(1,keepdims=True)
        out[i]=dl.mean(0)
    return out
if __name__=='__main__':
    w=W0
    m=np.where((ctl==0)&ok)[0]
    o=fd_table(w,X[m])
    for i in range(40):
        j=perm[i]
        if i<j:
            print(f'{N[i]:6s} {o[i]}   {N[j]:6s} {o[j]}   diff {np.abs(o[i]-o[j]).sum():.2f}')
