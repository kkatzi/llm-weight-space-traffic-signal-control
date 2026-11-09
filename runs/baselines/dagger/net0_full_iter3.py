"""DAgger distillation of Net 0, depth 8, iteration 3; 14567 labelled states"""


def decide(x):
    if x[5] <= 0.5022:  # q_E2
        if x[12] <= 0.5022:  # d_N0
            if x[2] <= 0.1674:  # q_N2
                if x[45] <= 0.6583:  # t_green
                    if x[42] <= 0.5000:  # phase2
                        if x[6] <= 0.5022:  # q_S0
                            if x[4] <= 0.1674:  # q_E1
                                if x[8] <= 0.5022:  # q_S2
                                    return 2
                                else:
                                    return 1
                            else:
                                if x[8] <= 0.8348:  # q_S2
                                    return 2
                                else:
                                    return 2
                        else:
                            if x[10] <= 0.1674:  # q_W1
                                if x[4] <= 0.5022:  # q_E1
                                    return 0
                                else:
                                    return 2
                            else:
                                if x[7] <= 0.8348:  # q_S1
                                    return 2
                                else:
                                    return 0
                    else:
                        if x[11] <= 0.1674:  # q_W2
                            if x[18] <= 0.5022:  # d_S0
                                if x[5] <= 0.1674:  # q_E2
                                    return 2
                                else:
                                    return 3
                            else:
                                if x[23] <= 0.1674:  # d_W2
                                    return 0
                                else:
                                    return 0
                        else:
                            if x[17] <= 0.1674:  # d_E2
                                if x[23] <= 0.5022:  # d_W2
                                    return 2
                                else:
                                    return 3
                            else:
                                if x[7] <= 0.5022:  # q_S1
                                    return 3
                                else:
                                    return 3
                else:
                    if x[45] <= 0.9083:  # t_green
                        if x[1] <= 0.1674:  # q_N1
                            return 2
                        else:
                            return 2
                    else:
                        return 2
            else:
                if x[3] <= 0.5022:  # q_E0
                    if x[11] <= 0.5022:  # q_W2
                        if x[6] <= 0.5022:  # q_S0
                            if x[14] <= 0.5022:  # d_N2
                                if x[3] <= 0.1674:  # q_E0
                                    return 1
                                else:
                                    return 2
                            else:
                                if x[17] <= 0.5022:  # d_E2
                                    return 1
                                else:
                                    return 1
                        else:
                            if x[8] <= 0.5022:  # q_S2
                                if x[13] <= 0.1674:  # d_N1
                                    return 0
                                else:
                                    return 0
                            else:
                                if x[13] <= 0.1674:  # d_N1
                                    return 1
                                else:
                                    return 0
                    else:
                        if x[8] <= 0.5022:  # q_S2
                            if x[5] <= 0.1674:  # q_E2
                                if x[2] <= 0.5022:  # q_N2
                                    return 3
                                else:
                                    return 1
                            else:
                                if x[10] <= 0.1674:  # q_W1
                                    return 3
                                else:
                                    return 3
                        else:
                            if x[11] <= 0.8348:  # q_W2
                                if x[1] <= 0.1674:  # q_N1
                                    return 1
                                else:
                                    return 1
                            else:
                                if x[5] <= 0.1674:  # q_E2
                                    return 1
                                else:
                                    return 3
                else:
                    if x[2] <= 0.8348:  # q_N2
                        if x[4] <= 0.5022:  # q_E1
                            if x[20] <= 0.5022:  # d_S2
                                if x[18] <= 0.8348:  # d_S0
                                    return 2
                                else:
                                    return 0
                            else:
                                if x[2] <= 0.5022:  # q_N2
                                    return 2
                                else:
                                    return 1
                        else:
                            if x[15] <= 0.8348:  # d_E0
                                if x[8] <= 0.5022:  # q_S2
                                    return 2
                                else:
                                    return 1
                            else:
                                if x[38] <= 0.0146:  # d_W0u
                                    return 2
                                else:
                                    return 2
                    else:
                        if x[4] <= 0.5022:  # q_E1
                            if x[8] <= 0.1674:  # q_S2
                                return 1
                            else:
                                if x[23] <= 0.8348:  # d_W2
                                    return 1
                                else:
                                    return 1
                        else:
                            if x[8] <= 0.8348:  # q_S2
                                if x[11] <= 0.1674:  # q_W2
                                    return 2
                                else:
                                    return 2
                            else:
                                if x[10] <= 0.8348:  # q_W1
                                    return 1
                                else:
                                    return 2
        else:
            if x[6] <= 0.5022:  # q_S0
                if x[4] <= 0.1674:  # q_E1
                    if x[23] <= 0.5022:  # d_W2
                        if x[2] <= 0.5022:  # q_N2
                            if x[13] <= 0.5022:  # d_N1
                                if x[23] <= 0.1674:  # d_W2
                                    return 0
                                else:
                                    return 0
                            else:
                                if x[21] <= 0.8348:  # d_W0
                                    return 0
                                else:
                                    return 0
                        else:
                            if x[0] <= 0.8348:  # q_N0
                                if x[7] <= 0.1674:  # q_S1
                                    return 1
                                else:
                                    return 0
                            else:
                                if x[8] <= 0.5022:  # q_S2
                                    return 0
                                else:
                                    return 1
                    else:
                        if x[8] <= 0.5022:  # q_S2
                            if x[42] <= 0.5000:  # phase2
                                return 0
                            else:
                                if x[1] <= 0.8348:  # q_N1
                                    return 3
                                else:
                                    return 0
                        else:
                            if x[1] <= 0.1674:  # q_N1
                                return 1
                            else:
                                if x[2] <= 0.5022:  # q_N2
                                    return 3
                                else:
                                    return 1
                else:
                    if x[14] <= 0.5022:  # d_N2
                        if x[10] <= 0.1674:  # q_W1
                            if x[1] <= 0.5022:  # q_N1
                                if x[15] <= 0.8348:  # d_E0
                                    return 0
                                else:
                                    return 2
                            else:
                                if x[41] <= 0.5000:  # phase1
                                    return 0
                                else:
                                    return 0
                        else:
                            if x[12] <= 0.8348:  # d_N0
                                if x[2] <= 0.1674:  # q_N2
                                    return 2
                                else:
                                    return 2
                            else:
                                if x[41] <= 0.5000:  # phase1
                                    return 0
                                else:
                                    return 2
                    else:
                        if x[0] <= 0.8348:  # q_N0
                            if x[8] <= 0.5022:  # q_S2
                                if x[16] <= 0.5022:  # d_E1
                                    return 1
                                else:
                                    return 2
                            else:
                                if x[3] <= 0.8348:  # q_E0
                                    return 1
                                else:
                                    return 1
                        else:
                            if x[2] <= 0.8348:  # q_N2
                                return 0
                            else:
                                if x[25] <= 0.0293:  # q_N1u
                                    return 0
                                else:
                                    return 1
            else:
                if x[41] <= 0.5000:  # phase1
                    if x[11] <= 0.8348:  # q_W2
                        if x[13] <= 0.5022:  # d_N1
                            if x[2] <= 0.8348:  # q_N2
                                if x[15] <= 0.8348:  # d_E0
                                    return 0
                                else:
                                    return 2
                            else:
                                if x[19] <= 0.8348:  # d_S1
                                    return 1
                                else:
                                    return 0
                        else:
                            if x[14] <= 0.8348:  # d_N2
                                if x[10] <= 0.8348:  # q_W1
                                    return 0
                                else:
                                    return 0
                            else:
                                if x[7] <= 0.1674:  # q_S1
                                    return 1
                                else:
                                    return 0
                    else:
                        if x[18] <= 0.8348:  # d_S0
                            if x[8] <= 0.5022:  # q_S2
                                if x[0] <= 0.8348:  # q_N0
                                    return 3
                                else:
                                    return 3
                            else:
                                return 3
                        else:
                            if x[0] <= 0.8348:  # q_N0
                                if x[15] <= 0.1674:  # d_E0
                                    return 3
                                else:
                                    return 0
                            else:
                                return 0
                else:
                    if x[21] <= 0.8348:  # d_W0
                        if x[22] <= 0.1674:  # d_W1
                            return 0
                        else:
                            if x[18] <= 0.8348:  # d_S0
                                return 0
                            else:
                                return 0
                    else:
                        if x[4] <= 0.5022:  # q_E1
                            if x[12] <= 0.8348:  # d_N0
                                return 2
                            else:
                                return 0
                        else:
                            if x[13] <= 0.8348:  # d_N1
                                return 2
                            else:
                                return 2
    else:
        if x[4] <= 0.1674:  # q_E1
            if x[11] <= 0.5022:  # q_W2
                if x[0] <= 0.8348:  # q_N0
                    if x[8] <= 0.5022:  # q_S2
                        if x[18] <= 0.8348:  # d_S0
                            if x[10] <= 0.1674:  # q_W1
                                if x[2] <= 0.8348:  # q_N2
                                    return 3
                                else:
                                    return 3
                            else:
                                if x[15] <= 0.5022:  # d_E0
                                    return 3
                                else:
                                    return 2
                        else:
                            if x[0] <= 0.5022:  # q_N0
                                if x[39] <= 0.0146:  # d_W1u
                                    return 0
                                else:
                                    return 3
                            else:
                                if x[2] <= 0.1674:  # q_N2
                                    return 0
                                else:
                                    return 0
                    else:
                        if x[2] <= 0.5022:  # q_N2
                            if x[7] <= 0.5022:  # q_S1
                                if x[17] <= 0.8348:  # d_E2
                                    return 1
                                else:
                                    return 3
                            else:
                                return 0
                        else:
                            if x[23] <= 0.1674:  # d_W2
                                return 1
                            else:
                                if x[17] <= 0.8348:  # d_E2
                                    return 1
                                else:
                                    return 1
                else:
                    if x[6] <= 0.5022:  # q_S0
                        if x[11] <= 0.1674:  # q_W2
                            return 0
                        else:
                            if x[13] <= 0.8348:  # d_N1
                                return 3
                            else:
                                return 3
                    else:
                        if x[23] <= 0.5022:  # d_W2
                            if x[13] <= 0.5022:  # d_N1
                                return 0
                            else:
                                return 0
                        else:
                            return 0
            else:
                if x[45] <= 0.6583:  # t_green
                    if x[19] <= 0.8348:  # d_S1
                        if x[18] <= 0.8348:  # d_S0
                            if x[9] <= 0.8348:  # q_W0
                                if x[9] <= 0.5022:  # q_W0
                                    return 3
                                else:
                                    return 3
                            else:
                                return 3
                        else:
                            if x[0] <= 0.8348:  # q_N0
                                return 3
                            else:
                                if x[11] <= 0.8348:  # q_W2
                                    return 0
                                else:
                                    return 3
                    else:
                        if x[14] <= 0.5022:  # d_N2
                            if x[13] <= 0.8348:  # d_N1
                                if x[13] <= 0.5022:  # d_N1
                                    return 3
                                else:
                                    return 0
                            else:
                                if x[34] <= 0.1025:  # d_E0u
                                    return 0
                                else:
                                    return 0
                        else:
                            return 3
                else:
                    return 3
        else:
            if x[14] <= 0.8348:  # d_N2
                if x[4] <= 0.5022:  # q_E1
                    if x[23] <= 0.5022:  # d_W2
                        if x[2] <= 0.1674:  # q_N2
                            if x[22] <= 0.1674:  # d_W1
                                if x[17] <= 0.8348:  # d_E2
                                    return 2
                                else:
                                    return 3
                            else:
                                if x[6] <= 0.5022:  # q_S0
                                    return 2
                                else:
                                    return 2
                        else:
                            if x[10] <= 0.1674:  # q_W1
                                return 3
                            else:
                                if x[3] <= 0.5022:  # q_E0
                                    return 3
                                else:
                                    return 2
                    else:
                        if x[21] <= 0.8348:  # d_W0
                            if x[17] <= 0.8348:  # d_E2
                                if x[23] <= 0.8348:  # d_W2
                                    return 2
                                else:
                                    return 3
                            else:
                                if x[15] <= 0.8348:  # d_E0
                                    return 3
                                else:
                                    return 3
                        else:
                            if x[8] <= 0.5022:  # q_S2
                                if x[22] <= 0.8348:  # d_W1
                                    return 2
                                else:
                                    return 2
                            else:
                                return 3
                else:
                    if x[10] <= 0.1674:  # q_W1
                        if x[11] <= 0.5022:  # q_W2
                            if x[20] <= 0.5022:  # d_S2
                                if x[6] <= 0.1674:  # q_S0
                                    return 2
                                else:
                                    return 2
                            else:
                                return 2
                        else:
                            if x[17] <= 0.8348:  # d_E2
                                return 2
                            else:
                                if x[38] <= 0.0732:  # d_W0u
                                    return 3
                                else:
                                    return 3
                    else:
                        if x[11] <= 0.8348:  # q_W2
                            if x[18] <= 0.8348:  # d_S0
                                if x[2] <= 0.5022:  # q_N2
                                    return 2
                                else:
                                    return 2
                            else:
                                return 2
                        else:
                            if x[10] <= 0.5022:  # q_W1
                                return 2
                            else:
                                if x[40] <= 0.5000:  # phase0
                                    return 2
                                else:
                                    return 2
            else:
                if x[20] <= 0.8348:  # d_S2
                    if x[10] <= 0.1674:  # q_W1
                        return 3
                    else:
                        if x[3] <= 0.8348:  # q_E0
                            return 1
                        else:
                            return 2
                else:
                    if x[10] <= 0.8348:  # q_W1
                        if x[11] <= 0.8348:  # q_W2
                            if x[16] <= 0.8348:  # d_E1
                                return 1
                            else:
                                return 1
                        else:
                            return 1
                    else:
                        return 2
