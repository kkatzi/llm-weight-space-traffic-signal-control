from common import *
from mirror import perm
for c in range(3):
    m=ctl==c
    print(S['controller_names'][c])
    mu=X[m].mean(0)
    for i in range(40):
        j=perm[i]
        if i<j: print(f'   {N[i]:6s} {mu[i]:.3f}  {N[j]:6s} {mu[j]:.3f}   ratio {mu[i]/max(mu[j],1e-6):.2f}')
