from common import *
from build_gate import build, counts
w=build(W0)
V=counts(X)
m=ok
Xm=X[m]
G=(31,0,59,54); R=(25,35,39,15)
h1o=np.tanh(Xm@W0['W1'].T+W0['b1']); h1n=np.tanh(Xm@w['W1'].T+w['b1'])
print('gate activation (+1 = phase empty) frequency over ok states:', ((h1n[:,list(G)])>0).mean(0), ' exact-ish saturation:', (np.abs(h1n[:,list(G)])>0.99).mean(0))
h2n=np.tanh(h1n@w['W2'].T+w['b2'])
print('relay activation freq', (h2n[:,list(R)]>0).mean(0), (np.abs(h2n[:,list(R)])>0.999).mean(0))
g0=tools.greedy(W0,Xm)
# variant: only layer-1 gate change but keep old W2/W3 for relays (i.e. measure L1 perturbation)
def fwd(W1,b1,W2,b2,W3,b3,Xa):
    return (np.tanh(np.tanh(Xa@W1.T+b1)@W2.T+b2)@W3.T+b3)
# L1 replaced, old downstream (with b2 comp)
l=fwd(w['W1'],w['b1'],W0['W2'],w['b2'],W0['W3'],W0['b3'],Xm)
print('L1 gate units only (old relays): change', (l.argmax(1)!=g0).mean())
# relays ablated only
w2={k:v.copy() for k,v in W0.items()}
w2['W2']=w['W2'].copy(); w2['W3']=w['W3'].copy(); w2['b3']=w['b3'].copy()
for p in range(4): w2['W2'][R[p],G[p]]=W0['W2'][R[p],G[p]]
# relay h2 from old net but W3 new -> relay contributes constant? emulate: set relay h2 to -1
h1=np.tanh(Xm@W0['W1'].T+W0['b1']); h2=np.tanh(h1@W0['W2'].T+W0['b2']); h2[:,list(R)]=-1
l=h2@w['W3'].T+w['b3']
print('relays ablated only: change',(l.argmax(1)!=g0).mean())
g1=tools.greedy(w,Xm)
ch=np.where(g1!=g0)[0]
Vm=V[m]
fix=(Vm[ch,g0[ch]]<0.5)
print('changed',len(ch),'fixes(empty old pick)',fix.sum())
print('side-effect changes: load (vehicles) of old pick vs new pick')
for k in ch[~fix][:40]:
    print('   cur',cur[m][k],'old',g0[k],'new',g1[k],'V',np.round(Vm[k]).astype(int),'Lq',np.round(Lq[m][k],1))
