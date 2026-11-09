import sys,tools
w=tools.load('start_weights.npz')
d=[float(x) for x in sys.argv[2:6]]
w['b3']=w['b3']+d
tools.save(w,sys.argv[1])
print(tools.edit_size(w,tools.load('start_weights.npz')))
