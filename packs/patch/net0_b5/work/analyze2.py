import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=200, precision=2, suppress=True)
w=tools.load('start_weights.npz')
S=tools.states()
X=S['obs'].astype(np.float64); A=S['act']; ep=S['episode']; T=S['time_s']; C=S['controller']
names=tools.names()
m=(C==0)&(ep==2)
Xe=X[m]; Ae=A[m]; Te=T[m]
ph=Xe[:,40:44].argmax(1)
# print phase timeline compressed
segs=[]
cur=ph[0]; st=Te[0]
for i in range(1,len(ph)):
    if ph[i]!=cur:
        segs.append((cur,st,Te[i]-st)); cur=ph[i]; st=Te[i]
segs.append((cur,st,Te[-1]-st))
print('num segs',len(segs))
print([ (int(a),int(b),int(c)) for a,b,c in segs[-40:]])
# print last part observations
for i in range(len(Xe)-200, len(Xe), 10):
    x=Xe[i]
    print(Te[i], 'ph',ph[i],'act',Ae[i],'ok',x[44],'tg',round(x[45],2),'q',x[0:12],'d',x[12:24], 'uq', x[24:32])
