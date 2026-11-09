import sys,tools
d=[float(v) for v in sys.argv[2:6]]
w=tools.load('start_weights.npz')
for i in range(4): w['b3'][i]+=d[i]
tools.save(w,sys.argv[1]);print(tools.edit_size(w,tools.load('start_weights.npz')))
