from common import *
# 180-degree rotation: N<->S, E<->W
perm=np.arange(46)
def sw(a,b):
    perm[a],perm[b]=b,a
for k in range(3):
    for off in (0,12):
        sw(off+0*3+k, off+2*3+k); sw(off+1*3+k, off+3*3+k)
for k in range(2):
    for off in (24,32):
        sw(off+0*2+k, off+2*2+k); sw(off+1*2+k, off+3*2+k)
assert sorted(perm)==list(range(46))
def mirror(Xa): return Xa[:,perm]
if __name__=='__main__':
    print([ (N[i],N[perm[i]]) for i in range(46) if perm[i]!=i][:8])
    w=W0
    for c in range(3):
        m=(ctl==c)&ok
        g1=tools.greedy(w,X[m]); g2=tools.greedy(w,mirror(X[m]))
        print(S['controller_names'][c],'mirror disagreement at ok', (g1!=g2).mean(), 'n',m.sum())
    # input weight columns compare
    W1=w['W1']
    for i in range(40):
        j=perm[i]
        if i<j:
            print(f'{N[i]:6s}/{N[j]:6s} col norm {np.linalg.norm(W1[:,i]):.2f} {np.linalg.norm(W1[:,j]):.2f}  corr {np.corrcoef(W1[:,i],W1[:,j])[0,1]:.2f}')
