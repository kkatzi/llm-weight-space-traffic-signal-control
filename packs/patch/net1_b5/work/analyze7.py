from common import *
w=W0
m=np.where((ctl==0)&ok)[0]
Xm=X[m]
a1=Xm@w['W1'].T+w['b1']; h1=np.tanh(a1); a2=h1@w['W2'].T+w['b2']; h2=np.tanh(a2)
# Jacobian d logits / d x: W3 diag(1-h2^2) W2 diag(1-h1^2) W1
J=np.einsum('ok,nk,kj,nj,ji->noi', w['W3'], 1-h2**2, w['W2'], 1-h1**2, w['W1'])
Jm=J.mean(0)  # 4 x 46
Jc=Jm-Jm.mean(0,keepdims=True)
print('centered mean sensitivity of logits to inputs (per unit of input in [0,1])')
for i,n in enumerate(N):
    print(f'{n:8s}', Jc[:,i])
