"""DAgger distillation of Net 2, depth 3, iteration 4; 17166 labelled states"""


def decide(x):
    if x[40] <= 0.5000:  # phase0
        if x[4] <= 0.1674:  # q_E1
            if x[11] <= 0.1674:  # q_W2
                return 0
            else:
                return 3
        else:
            if x[30] <= 0.3369:  # q_W0u
                return 2
            else:
                return 3
    else:
        if x[45] <= 0.2250:  # t_green
            if x[5] <= 0.5022:  # q_E2
                return 1
            else:
                return 3
        else:
            if x[36] <= 0.0146:  # d_S0u
                return 1
            else:
                return 1
