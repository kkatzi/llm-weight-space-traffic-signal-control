from common import *
from mirror import perm, mirror
from fd import fd_table
def copy_cols(w, pairs, biascomp=True, Xref=None):
    w={k:v.copy() for k,v in w.items()}
    Xref = X[ctl==0] if Xref is None else Xref
    for dst,src in pairs:
        i,j=ix[dst],ix[src]
        delta=w['W1'][:,j]-w['W1'][:,i]
        w['W1'][:,i]=w['W1'][:,j]
        if biascomp:
            w['b1']=w['b1']-delta*Xref[:,i].mean()
    return w
def report(w, label):
    print('==',label, tools.edit_size(w,W0))
    for c in range(3):
        m=(ctl==c)&ok
        g0=tools.greedy(W0,X[m]); g1=tools.greedy(w,X[m]); g1m=tools.greedy(w,mirror(X[m]))
        print(f'  {S["controller_names"][c]:14s} changed decisions at ok {(g0!=g1).mean():.3f}  mirror disagreement {(g1!=g1m).mean():.3f}')
    m=np.where((ctl==0)&ok)[0]
    o=fd_table(w,X[m])
    for nm in ['q_N1','q_S1','d_N1','d_S1','q_E0','q_W0','d_E0','d_W0','q_N0u','q_S0u']:
        print(f'   {nm:6s}', o[ix[nm]])
if __name__=='__main__':
    report(W0,'start')
    wA=copy_cols(W0,[('q_N1','q_S1'),('d_N1','d_S1'),('q_W0','q_E0')],biascomp=True)
    report(wA,'A biascomp')
    wA2=copy_cols(W0,[('q_N1','q_S1'),('d_N1','d_S1'),('q_W0','q_E0')],biascomp=False)
    report(wA2,'A no biascomp')
