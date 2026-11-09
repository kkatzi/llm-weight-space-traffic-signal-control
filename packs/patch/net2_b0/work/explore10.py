import numpy as np, sys
sys.path.insert(0, '.')
import tools
from collections import Counter
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64); act = S["act"]; ctl = S["controller"]; ep = S["episode"]; t = S["time_s"]
cap=22.4/7.5; capu=256/7.5
names = tools.names()
for c in range(3):
    mm=ctl==c
    hal = np.concatenate([X[mm,0:12]*cap, X[mm,24:32]*capu],1).mean(0)
    den = np.concatenate([X[mm,12:24]*cap, X[mm,32:40]*capu],1).mean(0)
    print(S["controller_names"][c])
    print("  halted:", " ".join(f"{n[2:]}:{v:.2f}" for n,v in zip(names[0:12]+names[24:32], hal)))
    print("  dens  :", " ".join(f"{n[2:]}:{v:.2f}" for n,v in zip(names[0:12]+names[24:32], den)))
    # spillback indicator: pocket full and upstream lane1 halted >0
    for ap,(pk,u1) in zip("NESW", [(2,25),(5,27),(8,29),(11,31)]):
        sp = (X[mm,pk]>0.99)&(X[mm,u1]>0)
        print(f"  {ap} pocket full & up1 halted: {sp.mean():.3f}", end="")
    print()
