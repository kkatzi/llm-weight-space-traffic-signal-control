import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=3, suppress=True)
S=tools.states(); X=S['obs'].astype(float); c=S['controller']; ep=S['episode']; t=S['time_s']; act=S['act']
w=tools.load('start_weights.npz')
print('W1 col 11 (q_W2):', w['W1'][:,11].round(2))
print('W1 col 23 (d_W2):', w['W1'][:,23].round(2))
print('W1 col 5 (q_E2):', w['W1'][:,5].round(2))
print('W1 col 17 (d_E2):', w['W1'][:,17].round(2))
print('col norms', np.linalg.norm(w['W1'],axis=0).round(2))
print('corr col11 vs col5', np.corrcoef(w['W1'][:,11],w['W1'][:,5])[0,1], 'col23 vs col17', np.corrcoef(w['W1'][:,23],w['W1'][:,17])[0,1])
# mirror pairs: N<->S, E<->W
def sub(X, q, d):
    Y=X.copy(); Y[:,11]=q; Y[:,23]=d; return Y
base=tools.greedy(w,X)
print('base==recorded act (start_network):', (base[c==0]==act[c==0]).mean())
variants={'mirror':sub(X,X[:,5],X[:,17]),'ones':sub(X,1,1),'half':sub(X,.33,.33)}
for n,Y in variants.items():
    a=tools.greedy(w,Y)
    for ci in range(3):
        m=c==ci
        ok=X[:,44]>0.5
        print(n, S['controller_names'][ci], 'changed', (a[m]!=base[m]).mean().round(3), 'dist', np.bincount(a[m&ok],minlength=4)/ (m&ok).sum(), 'base dist', np.bincount(base[m&ok],minlength=4)/(m&ok).sum())
# lock states
for e in [2,3,4,5]:
    m=(c==0)&(ep==e)&(t>4000)
    Y=X[m]
    for n,Z in [('fault',Y),('mirror',sub(Y,Y[:,5],Y[:,17])),('ones',sub(Y,1,1))]:
        lg=tools.forward(w,Z)
        print('ep',e,n,'acts',np.bincount(lg.argmax(1),minlength=4),'mean logits',lg.mean(0).round(2))
