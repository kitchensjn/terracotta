import numpy as np
from scipy.linalg import expm
import time
import numba 


rates = np.array([
    [
        [-0.01, 0.01],
        [0.001, -0.001]
    ],
    [
        [-0.1, 0.1],
        [2, -2]
    ]
])

times = [range(0, 1000, 100), range(0, 1000, 100)]

start = time.time()
all_transitions_final = []
for e in range(len(rates)):
    transitions = np.zeros((len(times[e]), 2, 2), dtype="float64")
    exponentiated = expm(rates[e])
    transitions[0] = np.linalg.matrix_power(exponentiated, times[e][0])
    for i in range(1, len(times[e])):
        for j in range(i-1, -1, -1):
            if times[e][j] > 0:
                power = times[e][i] / times[e][j]
                if power % 1 == 0:
                    transitions[i] = np.linalg.matrix_power(transitions[j], int(power))
                    break
            else:
                transitions[i] = np.linalg.matrix_power(exponentiated, times[e][i])
                break
            if j == 0:
                transitions[i] = np.linalg.matrix_power(exponentiated, times[e][i])
    all_transitions_final.append(transitions)
print(time.time() - start)

start = time.time()
all_transitions_new = []
for e in range(len(rates)):
    transitions = np.zeros((len(times[e]), 2, 2), dtype="float64")
    for t in range(len(times[e])):
        transitions[t] = expm(rates[e]*times[e][t])
    all_transitions_new.append(transitions)
print(time.time() - start)

start = time.time()
all_transitions_old = []
for e in range(len(rates)):
    transitions = np.zeros((len(times[e]), 2, 2), dtype="float64")
    for t in range(len(times[e])):
        transitions[t] = rates[e]*times[e][t]
    transitions = expm(transitions)
    all_transitions_old.append(transitions)
print(time.time() - start)


#print(all_transitions_old)
#print(all_transitions_final)

exit()





exponentiated = linalg.expm(transition_matrices[e])
transitions[0] = np.linalg.matrix_power(exponentiated, branch_lengths[e][0])
for i in range(1, len(times)):
    for j in range(i-1, -1, -1):
        if branch_lengths[e][j] > 0:
            power = branch_lengths[e][i] / branch_lengths[e][j]
            if power % 1 == 0:
                transitions[i] = np.linalg.matrix_power(transitions[j], int(power))
                break
        else:
            transitions[i] = np.linalg.matrix_power(exponentiated, branch_lengths[e][i])
            break
        if j == 0:
            transitions[i] = np.linalg.matrix_power(exponentiated, branch_lengths[e][i])