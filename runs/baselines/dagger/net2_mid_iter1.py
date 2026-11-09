"""DAgger distillation of Net 2, depth 5, iteration 1; 8054 labelled states"""


def decide(x):
    if x[10] <= 0.1674:  # q_W1
        if x[5] <= 0.1674:  # q_E2
            if x[2] <= 0.1674:  # q_N2
                if x[40] <= 0.5000:  # phase0
                    if x[15] <= 0.1674:  # d_E0
                        return 0
                    else:
                        return 0
                else:
                    if x[11] <= 0.5022:  # q_W2
                        return 1
                    else:
                        return 3
            else:
                if x[13] <= 0.1674:  # d_N1
                    if x[8] <= 0.5022:  # q_S2
                        return 1
                    else:
                        return 1
                else:
                    if x[6] <= 0.5022:  # q_S0
                        return 0
                    else:
                        return 0
        else:
            if x[6] <= 0.5022:  # q_S0
                if x[14] <= 0.5022:  # d_N2
                    if x[23] <= 0.1674:  # d_W2
                        return 3
                    else:
                        return 3
                else:
                    if x[11] <= 0.5022:  # q_W2
                        return 1
                    else:
                        return 3
            else:
                if x[1] <= 0.5022:  # q_N1
                    if x[17] <= 0.8348:  # d_E2
                        return 0
                    else:
                        return 3
                else:
                    if x[27] <= 0.0732:  # q_E1u
                        return 0
                    else:
                        return 3
    else:
        if x[2] <= 0.5022:  # q_N2
            if x[4] <= 0.1674:  # q_E1
                if x[17] <= 0.5022:  # d_E2
                    if x[1] <= 0.5022:  # q_N1
                        return 2
                    else:
                        return 0
                else:
                    if x[12] <= 0.5022:  # d_N0
                        return 3
                    else:
                        return 2
            else:
                if x[23] <= 0.8348:  # d_W2
                    if x[8] <= 0.5022:  # q_S2
                        return 2
                    else:
                        return 2
                else:
                    if x[10] <= 0.5022:  # q_W1
                        return 3
                    else:
                        return 2
        else:
            if x[20] <= 0.5022:  # d_S2
                if x[23] <= 0.5022:  # d_W2
                    if x[21] <= 0.5022:  # d_W0
                        return 1
                    else:
                        return 2
                else:
                    if x[16] <= 0.8348:  # d_E1
                        return 3
                    else:
                        return 2
            else:
                if x[1] <= 0.5022:  # q_N1
                    if x[5] <= 0.5022:  # q_E2
                        return 1
                    else:
                        return 3
                else:
                    if x[6] <= 0.5022:  # q_S0
                        return 2
                    else:
                        return 0
