# Notes
Edit: b3 (output biases) shifted by [0,-1,0,-1], so phases 1 and 3 (left-turn) are chosen a little less often. 2 parameters changed.
Why: the network mostly cycles ~15 s greens; left phases serve the least demand (30% turns, pocket of ~3 cars), so I tried making them slightly less attractive. About 3% of switch decisions change.
Checks: forward passes on recorded states for decision-change rates; 3 dev sim calls (of 10 used).
- A (b3 +[0,-1,1,0]): 45.0 dev (worse than 44.7 baseline)
- B (b3 +[0,-1,0,-1]): 44.6 dev (low 34.6, nominal 44.7, high 54.5) -> delivered
- C (b3 +[0,-1.5,0,-1.5]): 44.9 dev
Expectation: B's gain over the start net (0.1 veh-h) is within noise; the held-out result is likely about equal to the original. Risk: starving left pockets under heavier left demand.
Scripts: make.py
