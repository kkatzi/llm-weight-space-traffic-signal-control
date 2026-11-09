from common import *
w=W0
M=ok
Xm=X[M]
h1=np.tanh(Xm@w['W1'].T+w['b1']); a2=h1@w['W2'].T+w['b2']; h2=np.tanh(a2); lg=h2@w['W3'].T+w['b3']; g0=lg.argmax(1)
X0=X[ctl==0]; H1=np.tanh(X0@w['W1'].T+w['b1']); H2=np.tanh(np.tanh(X0@w['W1'].T+w['b1'])@w['W2'].T+w['b2'])
m1=H1.mean(0); m2=H2.mean(0)
res1=[]
for i in range(64):
    # replace unit i by constant -1 with b2 compensation to its mean
    hh=h1.copy(); hh[:,i]=-1
    b2c=w['b2']+w['W2'][:,i]*(m1[i]+1)
    l=np.tanh(hh@w['W2'].T+b2c)@w['W3'].T+w['b3']
    hh2=h1.copy(); hh2[:,i]=-1
    l2=np.tanh(hh2@w['W2'].T+w['b2'])@w['W3'].T+w['b3']
    res1.append((i,(l.argmax(1)!=g0).mean(),(l2.argmax(1)!=g0).mean(),np.abs(w['W2'][:,i]).sum(),H1[:,i].std(), np.abs(w['W1'][i]).sum()))
res1.sort(key=lambda r:r[1])
print('layer1: unit, change(comp), change(nocomp), sum|W2 col|, std, sum|W1 row|')
for r in res1[:16]: print('  ',r[0],round(r[1],4),round(r[2],4),round(r[3],2),round(r[4],3),round(r[5],2))
res2=[]
for k in range(64):
    hh=h2.copy(); hh[:,k]=-1
    b3c=w['b3']+w['W3'][:,k]*(m2[k]+1)
    l=hh@w['W3'].T+b3c
    res2.append((k,(l.argmax(1)!=g0).mean(),np.abs(w['W3'][:,k]).sum(),H2[:,k].std(),np.abs(w['W2'][k]).sum()))
res2.sort(key=lambda r:r[1])
print('layer2: unit, change(comp), sum|W3 col|, std, sum|W2 row|')
for r in res2[:16]: print('  ',r[0],round(r[1],4),round(r[2],2),round(r[3],3),round(r[4],2))
