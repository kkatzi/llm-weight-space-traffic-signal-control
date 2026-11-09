from common import *
from build_gate import build
from repairA import copy_cols
m=ok
g0=tools.greedy(W0,X[m])
for G in [(31,0,59,54),(31,0,27,43),(0,31,43,27),(31,0,5,4)]:
    for R in [(25,35,39,15),(25,35,39,9),(25,35,39,54)]:
        w=build(W0,G=G,R=R)
        print(G,R,'change at ok',round((tools.greedy(w,X[m])!=g0).mean(),4),'own',round((tools.greedy(w,X[m&(ctl==0)])!=tools.greedy(W0,X[m&(ctl==0)])).mean(),4), tools.edit_size(w,W0)['changed'])
