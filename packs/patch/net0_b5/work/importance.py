import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=3, suppress=True)
w=tools.load('start_weights.npz')
S=tools.states()
X=S['obs'].astype(np.float64); C=S['controller']; ep=S['episode']; T=S['time_s']
ok=X[:,44]>0.5
dead=(C==0)&(ep==2)&(T>3620)
Xo=X[ok&~dead]
g0=tools.greedy(w,Xo)
h1,h2=tools.hidden(w,Xo)
mu1=h1.mean(0); sd1=h1.std(0)
mu2=h2.mean(0); sd2=h2.std(0)
# layer-1 ablation: replace h1_j by mean (with b2 compensation implicitly by mean)
r1=[]
for j in range(64):
    H=h1.copy(); H[:,j]=mu1[j]
    lg=np.tanh(H@w['W2'].T+w['b2'])@w['W3'].T+w['b3']
    r1.append((lg.argmax(1)!=g0).mean())
# layer-1: replace by constant -1 without compensation
r1n=[]
for j in range(64):
    H=h1.copy(); H[:,j]=-1
    lg=np.tanh(H@w['W2'].T+w['b2'])@w['W3'].T+w['b3']
    r1n.append((lg.argmax(1)!=g0).mean())
r2=[]
for k in range(64):
    H=h2.copy(); H[:,k]=mu2[k]
    lg=H@w['W3'].T+w['b3']
    r2.append((lg.argmax(1)!=g0).mean())
r1=np.array(r1); r1n=np.array(r1n); r2=np.array(r2)
o1=np.argsort(r1); o2=np.argsort(r2)
print('n ok states',len(Xo))
print('L1 best (mean-ablation):',[(int(j),round(r1[j],4),round(r1n[j],4),round(sd1[j],2),round(mu1[j],2)) for j in o1[:12]])
print('L2 best (mean-ablation):',[(int(k),round(r2[k],4),round(sd2[k],2),round(mu2[k],2)) for k in o2[:12]])
np.save('work/imp_r1.npy',r1); np.save('work/imp_r2.npy',r2)
