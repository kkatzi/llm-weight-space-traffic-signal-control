from common import *
from build_gate import build, counts
from repairA import copy_cols
from deadlock import check
from mirror import mirror
def evaluate(w,label):
    print('=====',label,tools.edit_size(w,W0))
    V=counts(X)
    for c in range(3):
        m=(ctl==c)&ok
        g0=tools.greedy(W0,X[m]); g1=tools.greedy(w,X[m])
        ch=g0!=g1
        Vm=V[m]
        emp_old=(Vm[np.arange(m.sum()),g0]<0.5)&(Vm.max(1)>0.5)
        emp_new=(Vm[np.arange(m.sum()),g1]<0.5)&(Vm.max(1)>0.5)
        print(f'  {S["controller_names"][c]:14s} changed {ch.mean():.4f} ({ch.sum()})  picks-empty old {emp_old.sum()} new {emp_new.sum()}  mirror-dis {(g1!=tools.greedy(w,mirror(X[m]))).mean():.3f}')
    # all-states (incl not ok) change
    print('  all states change', (tools.greedy(W0,X)!=tools.greedy(w,X)).mean())
    check(w,label)
    print('  trainability',tools.trainability(w))
if __name__=='__main__':
    wg=build(W0)
    evaluate(wg,'gate only')
    wA=copy_cols(W0,[('q_N1','q_S1'),('d_N1','d_S1'),('q_W0','q_E0')],biascomp=False)
    wAg=build(wA)
    evaluate(wAg,'A + gate')
