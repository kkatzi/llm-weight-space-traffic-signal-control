"""DAgger distillation of Net 0, depth 3, iteration 5; 24163 labelled states"""


def decide(x):
    if x[2] <= 0.1674:  # q_N2
        if x[1] <= 0.5022:  # q_N1
            if x[7] <= 0.5022:  # q_S1
                return 2
            else:
                return 0
        else:
            if x[5] <= 0.5022:  # q_E2
                return 0
            else:
                return 3
    else:
        if x[1] <= 0.1674:  # q_N1
            if x[8] <= 0.5022:  # q_S2
                return 3
            else:
                return 1
        else:
            if x[7] <= 0.5022:  # q_S1
                return 1
            else:
                return 0
