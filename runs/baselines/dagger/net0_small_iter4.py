"""DAgger distillation of Net 0, depth 3, iteration 4; 21070 labelled states"""


def decide(x):
    if x[2] <= 0.5022:  # q_N2
        if x[6] <= 0.5022:  # q_S0
            if x[5] <= 0.1674:  # q_E2
                return 2
            else:
                return 3
        else:
            if x[17] <= 0.5022:  # d_E2
                return 0
            else:
                return 3
    else:
        if x[13] <= 0.5022:  # d_N1
            if x[5] <= 0.5022:  # q_E2
                return 1
            else:
                return 3
        else:
            if x[17] <= 0.8348:  # d_E2
                return 0
            else:
                return 3
