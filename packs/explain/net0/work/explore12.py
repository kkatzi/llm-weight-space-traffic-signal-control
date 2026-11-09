import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from evalh import *
import itertools
np.set_printoptions(precision=2, suppress=True, linewidth=250)
lanes = {"N0":0,"N1":1,"N2":2,"S0":6,"S1":7,"S2":8,"E2":5,"W2":11}
keys = list(lanes)
V=1/2.985
for tg in [12/60, 1.0]:
  res = {}
  for combo in itertools.product([0,1],repeat=8):
    x=np.zeros(46); x[42]=1; x[44]=1; x[45]=tg
    for k,v in zip(keys,combo):
        x[lanes[k]]=v*V; x[12+lanes[k]]=v*V
    l=tools.forward(w,x)[0]
    res[combo]=(l.argmax(), l)
  keep=[c for c in res if res[c][0]==2]
  print("tg",tg,"n keep",len(keep))
  for c in keep:
    print("  ", {k:v for k,v in zip(keys,c) if v}, np.round(res[c][1],2))
