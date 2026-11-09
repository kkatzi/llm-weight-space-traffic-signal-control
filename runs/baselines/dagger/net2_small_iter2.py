"""DAgger distillation of Net 2, depth 3, iteration 2; 11384 labelled states"""


def decide(x):
    if x[4] <= 0.1674:  # q_E1
        if x[11] <= 0.1674:  # q_W2
            if x[45] <= 0.5167:  # t_green
                return 0
            else:
                return 0
        else:
            if x[7] <= 0.5022:  # q_S1
                return 3
            else:
                return 0
    else:
        if x[23] <= 0.8348:  # d_W2
            if x[2] <= 0.5022:  # q_N2
                return 2
            else:
                return 1
        else:
            if x[22] <= 0.5022:  # d_W1
                return 3
            else:
                return 2
