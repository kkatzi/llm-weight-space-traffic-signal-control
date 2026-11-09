import numpy as np, sys
sys.path.insert(0, '.')
import tools
from collections import Counter
exec(open('work/explore6.py').read().split('m = (ctl==0)&ok&(a==cur)')[0])
QL = {0:[0,1,6,7],1:[2,8],2:[3,4,9,10],3:[5,11]}
dd = np.stack([ (X[:,[i+12 for i in QL[p]]]*cap).round().sum(1) for p in range(4)],1)
mm = (ctl==0)&ok
idx=np.where(mm)[0]
emp = dd[idx, a[idx]]==0
oth = np.array([sum(dd[i,p] for p in range(4) if p!=a[i]) for i in idx])
print("stays:", (a[idx]==cur[idx]).sum(), " stays with own empty:", ((a[idx]==cur[idx])&emp).sum(), " with others nonempty:", ((a[idx]==cur[idx])&emp&(oth>0)).sum())
print("switch to empty:", ((a[idx]!=cur[idx])&emp).sum(), " with others nonempty:", ((a[idx]!=cur[idx])&emp&(oth>0)).sum())
for i in idx[(a[idx]!=cur[idx])&emp&(oth>0)]: show(i)
print("--- stays with own empty & others nonempty")
for i in idx[(a[idx]==cur[idx])&emp&(oth>0)]: show(i)
print("--- stays with own pocket nonempty")
for i in idx[(a[idx]==cur[idx])&~emp]: show(i)
