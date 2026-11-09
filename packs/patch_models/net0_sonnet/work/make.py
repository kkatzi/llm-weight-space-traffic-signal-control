import sys,numpy as np
sys.path.insert(0,'.')
import tools
w=tools.load('start_weights.npz')
for name,d in [('A',(0,-1,1,0)),('B',(0,-1,0,-1))]:
    v=dict(w);v['b3']=w['b3']+np.array(d);tools.save(v,f'work/{name}.npz');print(name,tools.edit_size(v,w))
