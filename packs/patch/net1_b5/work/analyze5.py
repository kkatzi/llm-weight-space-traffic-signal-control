from common import *
g=tools.greedy(W0,X)
m=np.where((ctl==0)&ok)[0]
am=Lq[m].argmax(1)
dev=m[g[m]!=am]
print('deviations',len(dev),'of',len(m))
T=np.zeros((4,4),int)
for a,b in zip(Lq[dev].argmax(1),g[dev]): T[a,b]+=1
print('rows=argmax halting, cols=net pick\n',T)
# in deviations, is the argmax-halting phase the current phase?
print('argmax halting == current in deviations', (Lq[dev].argmax(1)==cur[dev]).mean())
# deviations excluding current
L2=Lq.copy(); L2[np.arange(len(X)),cur]=-1
am2=L2[m].argmax(1)
print('net == argmax halting excluding current', (g[m]==am2).mean())
dev2=m[(g[m]!=am2)&(g[m]!=cur[m])]
print('n dev2',len(dev2))
for i in dev2[:25]:
    print(t[i],'cur',cur[i],'pick',g[i],'Lq',Lq[i],'Ld',Ld[i])
