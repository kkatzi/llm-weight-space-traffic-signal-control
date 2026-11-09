import numpy as np, sys
sys.path.insert(0, '.')
import tools
exec(open('work/explore6.py').read().split('m = (ctl==0)&ok&(a==cur)')[0])
nsE = (X[:,14]==0)&(X[:,20]==0)
for c in [0,2]:
    print(S["controller_names"][c])
    for i in np.where(ok&(ctl==c)&nsE&(a==1))[0]: show(i)
