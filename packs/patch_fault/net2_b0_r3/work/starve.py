"""Starvation events in the recorded episodes: runs of decisions where a stop-line lane has a halting vehicle
and its serving phase is not green."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
S = tools.states(); X = S["obs"].astype(np.float64); c = S["controller"]; ep = S["episode"]; t = S["time_s"]
nm = tools.names()
SERVE = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 0, 7: 0, 8: 1, 9: 2, 10: 2, 11: 3}
for ci, cn in enumerate(S["controller_names"]):
    tot_wait = 0; events = []
    for e in np.unique(ep[c == ci]):
        idx = np.where((c == ci) & (ep == e))[0]
        ph = X[idx, 40:44].argmax(1)
        for lane in range(11):
            run = 0; start = None
            for k, i in enumerate(idx):
                if X[i, lane] > 0 and ph[k] != SERVE[lane]:
                    if run == 0: start = t[i]
                    run += 1
                else:
                    if run * 5 >= 150: events.append((e, nm[lane], start, run * 5))
                    run = 0
            if run * 5 >= 150: events.append((e, nm[lane], start, run * 5))
    print(cn, "starvation runs >= 150 s:", len(events))
    for ev in events: print("   ", ev)
