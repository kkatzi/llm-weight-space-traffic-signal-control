import numpy as np, sys
sys.path.insert(0, '.')
import tools
from collections import Counter
exec(open('work/explore6.py').read().split('m = (ctl==0)&ok&(a==cur)')[0])
QL = {0:[0,1,6,7],1:[2,8],2:[3,4,9,10],3:[5,11]}
qq = np.stack([ (X[:,QL[p]]*cap).round().sum(1) for p in range(4)],1)
uq = (X[:,24:32]*capu).round()
mm = (ctl==0)&ok
idx=np.where(mm)[0]
sw = idx[a[idx]!=cur[idx]]
gap = np.array([max(qq[i,p] for p in range(4) if p!=cur[i]) - qq[i,a[i]] for i in sw])
print(Counter(gap))
# which phase was the max when the gap was large
big = sw[gap>=3]
print("n big", len(big))
print("chosen phase", Counter(a[big]), "max phase", Counter([max([p for p in range(4) if p!=cur[i]], key=lambda p: qq[i,p]) for i in big]))
for i in big[:30]: show(i)
