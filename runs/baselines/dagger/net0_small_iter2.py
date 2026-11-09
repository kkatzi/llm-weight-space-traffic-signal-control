"""DAgger distillation of Net 0, depth 3, iteration 2; 13290 labelled states"""


def decide(x):
    if x[0] <= 0.8348:  # q_N0
        if x[5] <= 0.1674:  # q_E2
            if x[7] <= 0.5022:  # q_S1
                return 2
            else:
                return 0
        else:
            if x[4] <= 0.1674:  # q_E1
                return 3
            else:
                return 2
    else:
        if x[5] <= 0.5022:  # q_E2
            if x[41] <= 0.5000:  # phase1
                return 0
            else:
                return 2
        else:
            if x[32] <= 0.6885:  # d_N0u
                return 0
            else:
                return 3
