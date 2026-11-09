import sys; sys.path.insert(0, '/home/traffic-ntua/wsp_v3_agent_runs/patch_models/net1_haiku'); sys.path.insert(0, '/home/traffic-ntua/wsp_v3_agent_runs/patch_models/net1_haiku/work')
import tools
from build_stay import build, W
for beta in [1.0, 2.0]:
    w = build(beta)
    p = tools.save(w, f"/home/traffic-ntua/wsp_v3_agent_runs/patch_models/net1_haiku/work/stay_b{beta:g}.npz")
    print(p, tools.edit_size(w, W))
