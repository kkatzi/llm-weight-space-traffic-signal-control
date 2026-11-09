import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=200, precision=3, suppress=True)
w=tools.load('start_weights.npz')
S=tools.states()
X=S['obs'].astype(np.float64); A=S['act']; ep=S['episode']; T=S['time_s']; C=S['controller']
def segs_of(m):
    Xe=X[m]; Te=T[m]; ph=Xe[:,40:44].argmax(1)
    out=[]; cur=ph[0]; st=Te[0]
    for i in range(1,len(ph)):
        if ph[i]!=cur: out.append((cur,Te[i]-st)); cur=ph[i]; st=Te[i]
    return out
for c in range(3):
    allseg=[]
    for e in range(6):
        m=(C==c)&(ep==e)&(T<3600)
        allseg+=segs_of(m)
    allseg=np.array(allseg)
    print(S['controller_names'][c], 'nseg',len(allseg))
    for p in range(4):
        d=allseg[allseg[:,0]==p,1]
        print('  phase',p,'count',len(d),'mean dur',d.mean().round(1),'quantiles',np.percentile(d,[10,50,90]),'max',d.max())
# transitions for network
c=0
trans=np.zeros((4,4),int)
for e in range(6):
    m=(C==c)&(ep==e)
    s=segs_of(m)
    for a,b in zip(s[:-1],s[1:]): trans[a[0],b[0]]+=1
print('network transitions\n',trans)
# Decisions at ok=1 on network states: stay vs switch
m=(C==0)&(X[:,44]>0.5)
ph=X[m,40:44].argmax(1); a=A[m]
print('ok states', m.sum(), 'stay frac', (ph==a).mean())
for p in range(4):
    mm=ph==p
    print('phase',p,'n',mm.sum(),'choice dist',np.bincount(a[mm],minlength=4))
