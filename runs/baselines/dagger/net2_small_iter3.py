"""DAgger distillation of Net 2, depth 3, iteration 3; 15350 labelled states"""


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
        if x[35] <= 0.0146:  # d_E1u
            if x[16] <= 0.1674:  # d_E1
                return 1
            else:
                return 1
        else:
            if x[25] <= 0.7178:  # q_N1u
                return 1
            else:
                return 1
