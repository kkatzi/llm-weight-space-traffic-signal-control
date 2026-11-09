import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from sparsecheck import *
np.set_printoptions(precision=2, suppress=True, linewidth=250)
# cur=2, tg=1: decision table over (nsT halted spread N0,N1,S0,S1 ; nsL spread N2,S2 ; E2 ; W2)
def mkx(nsT=0, nsL=0, e2=0, w2=0, ewT=0, tg=1.0, cur=2):
    x=np.zeros(46); x[40+cur]=1; x[44]=1; x[45]=tg
    order=["N0","S0","N1","S1","N0","S0","N1","S1","N0","S0","N1","S1"]
    cnt={}
    for k in order[:nsT]: cnt[k]=cnt.get(k,0)+1
    for k in ["N2","S2","N2","S2","N2","S2"][:nsL]: cnt[k]=cnt.get(k,0)+1
    cnt["E2"]=e2; cnt["W2"]=w2
    for k in ["E0","W1","E1","W0","E0","W1","E1","W0"][:ewT]: cnt[k]=cnt.get(k,0)+1
    for k,v in cnt.items():
        x[LANES[k]]=v/2.985; x[12+LANES[k]]=v/2.985
    return x
for tg in [0.2, 1.0]:
  print("tg",tg, " rows: nsT 0..5 ; cols: (nsL,E2,W2)")
  cols=[(a,b,c) for a in range(3) for b in range(2) for c in range(3)]
  print("      "+" ".join(f"{a}{b}{c}" for a,b,c in cols))
  for n in range(6):
    row=[]
    for a,b,c in cols:
        row.append(str(tools.greedy(w, mkx(n,a,b,c,tg=tg))[0]))
    print(f"nsT={n} "+"   ".join(row))
