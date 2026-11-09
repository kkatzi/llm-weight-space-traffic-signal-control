import numpy as np, tools
N = tools.names(); idx = {n:k for k,n in enumerate(N)}
w0 = tools.load("start_weights.npz"); w = tools.load("work/weights.npz")
print("edit size", tools.edit_size(w, w0))
S = tools.states(); X, ep, t, c = S["obs"], S["episode"], S["time_s"], S["controller"]
# the recorded stuck state
m = (c==0)&(ep==2)&(t>4000)
x = X[m][0]; print("stuck state old pick", tools.greedy(w0,x), "new pick", tools.greedy(w,x), "logits new", tools.forward(w,x).round(2))
def mk(phase, tg, lanes_q, upstream=None):
    x = np.zeros(46); x[40+phase]=1; x[44]=1.0 if tg>=10 else 0.0; x[45]=min(tg/60,1)
    for n,v in lanes_q.items(): x[idx["q_"+n]]=v; x[idx["d_"+n]]=v
    if upstream:
        for n,v in upstream.items(): x[idx["d_"+n]]=v
    return x
tests = {
 "ph2 ok, 1 car N0": mk(2,12,{"N0":1/3}),
 "ph2 ok, 1 car W2": mk(2,12,{"W2":1/3}),
 "ph2 ok, 1 car S1": mk(2,12,{"S1":1/3}),
 "ph2 ok, 1 car N0 + 1 car E0 stopline": mk(2,12,{"N0":1/3,"E0":1/3}),
 "ph2 ok, 1 car N0 + 1 upstream E0u": mk(2,12,{"N0":1/3},{"E0u":1/34}),
 "ph2 ok, 2 cars N0 + 1 upstream E0u": mk(2,12,{"N0":2/3},{"E0u":1/34}),
 "ph2 ok, 1 car N0 + 3 upstream E0u": mk(2,12,{"N0":1/3},{"E0u":3/34}),
 "ph2 not ok (tg 7), 1 car N0": mk(2,7,{"N0":1/3}),
 "ph2 ok, empty": mk(2,12,{}),
 "ph0 ok, 1 car W2": mk(0,12,{"W2":1/3}),
 "ph3 ok, 1 car N0": mk(3,12,{"N0":1/3}),
 "ph2 ok, 1 car N0 + 1 car E2 (left)": mk(2,12,{"N0":1/3,"E2":1/3}),
}
for k,x in tests.items():
    print(f"{k:45s} old {tools.greedy(w0,x)[0]} new {tools.greedy(w,x)[0]}  new logits {tools.forward(w,x).round(1)}")
