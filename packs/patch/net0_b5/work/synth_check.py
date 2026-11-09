import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
from build import build, demand, SL, UP
np.set_printoptions(linewidth=220, precision=3, suppress=True)
rng=np.random.default_rng(0)
w0=tools.load('start_weights.npz'); w=tools.load('work/v1.npz' if os.path.exists('work/v1.npz') else 'v1.npz')
K=(37,0,11,5)
def synth(n, p_occ):
    X=np.zeros((n,46))
    cs=22.4/7.5; cu=256/7.5
    occ=rng.random((n,12))<p_occ
    nveh=rng.integers(1,4,(n,12))*occ
    X[:,12:24]=np.minimum(nveh/cs,1); X[:,0:12]=np.minimum(rng.integers(0,4,(n,12))*occ,nveh)/cs
    uocc=rng.random((n,8))<p_occ
    unv=rng.integers(1,35,(n,8))*uocc
    X[:,32:40]=np.minimum(unv/cu,1); X[:,24:32]=np.minimum(rng.integers(0,35,(n,8))*uocc,unv)/cu
    ph=rng.integers(0,4,n); X[np.arange(n),40+ph]=1
    X[:,44]=rng.random(n)<0.7; X[:,45]=rng.random(n)
    return X
for p_occ in [0.05,0.15,0.4,0.8]:
    X=synth(20000,p_occ)
    E=demand(X)==0
    h1,h2=tools.hidden(w,X)
    bad=0
    for p,k in enumerate(K):
        bad+=((h2[:,k]>0.99)!=E[:,p]).sum()+((np.abs(h2[:,k])<0.99)).sum()
    g=tools.greedy(w,X)
    some=E.any(1)&~E.all(1)
    print(f'p_occ {p_occ}: flag errors {bad}, states with some empty {some.sum()}, chosen nonempty there {(~E[some][np.arange(some.sum()),g[some]]).mean():.4f}, min |h2 flag| {np.abs(h2[:,list(K)]).min():.4f}')
# drain-like: only stop-line vehicles, no upstream
X=synth(20000,0.15); X[:,24:40]=0
E=demand(X)==0; g=tools.greedy(w,X); some=E.any(1)&~E.all(1)
print('drain-like: chosen nonempty', (~E[some][np.arange(some.sum()),g[some]]).mean(), 'n',some.sum())
