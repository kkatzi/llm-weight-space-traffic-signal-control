# Notes
Built: start network with b3[2] (logit bias of phase 2, EW_through) raised by +1.0. One parameter changed (work/mk.py).
Why: the network agrees 100% with its recorded decisions and is confident, so only small, safe nudges were tried. Phase 2 serves the
heaviest demand (main road through traffic).
Checks (dev, 10-call budget, 4 used): start 44.5; b3[1]-1: 44.6; b3[2]+1: 44.3 (final); b3=[+.5,-.5,+1.5,0]: 44.6.
All differences are within seed noise (~0.3); the gain is not demonstrated. Expect the test result to be about equal to the original.
Risk: slight over-serving of EW_through at the expense of NS phases under heavy NS queues.
