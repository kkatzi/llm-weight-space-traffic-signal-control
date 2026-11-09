import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
from build import demand
S=tools.states(); w0=tools.load('start_weights.npz')
X=S['obs'].astype(np.float64); ok=X[:,44]>0.5; E=demand(X)==0; norm=ok&~E.any(1)
Xn=X[norm]; g0=tools.greedy(w0,Xn)
h1,h2=tools.hidden(w0,Xn)
Xall=X; h1a,h2a=tools.hidden(w0,Xall); mu1=h1a.mean(0); mu2=h2a.mean(0)
def agree(J,K):
    H=h1.copy(); H[:,list(J)]=mu1[list(J)]
    H2=np.tanh(H@w0['W2'].T+w0['b2']); H2[:,list(K)]=mu2[list(K)]
    return ((H2@w0['W3'].T+w0['b3']).argmax(1)==g0).mean()
r1=np.load('imp_r1.npy'); r2=np.load('imp_r2.npy')
print('J only', agree((60,57,32,13),()), 'K only', agree((),(37,0,11,5)), 'both', agree((60,57,32,13),(37,0,11,5)))
o1=np.argsort(r1)[:8]; o2=np.argsort(r2)[:8]
print('top8 L1',o1, r1[o1].round(4)); print('top8 L2',o2, r2[o2].round(4))
