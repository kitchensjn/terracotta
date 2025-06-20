import numpy as np
from scipy import linalg
from scipy.special import logsumexp


# In natural space
Q = np.array([[-0.1, 0.1],[0.1, -0.1]])
M02 = np.matmul(np.array([[1, 0]]), linalg.expm(Q*1))
M13 = np.matmul(np.array([[0, 1]]), linalg.expm(Q*2))

## If 2 is in deme 0
N2_0_M23 = np.matmul(np.array([[1, 0]]), linalg.expm(Q*1))
N2_0_M24 = np.matmul(np.array([[1, 0]]), linalg.expm(Q*2))
N2_0_M34 = np.matmul(np.multiply(M13, N2_0_M23), linalg.expm(Q*1))
N2_0_N4 = np.multiply(N2_0_M24, N2_0_M34)
N2_0_rootlike = np.sum(N2_0_N4)

## If 2 is in deme 1
N2_1_M23 = np.matmul(np.array([[0, 1]]), linalg.expm(Q*1))
N2_1_M24 = np.matmul(np.array([[0, 1]]), linalg.expm(Q*2))
N2_1_M34 = np.matmul(np.multiply(M13, N2_1_M23), linalg.expm(Q*1))
N2_1_N4 = np.multiply(N2_1_M24, N2_1_M34)
N2_1_rootlike = np.sum(N2_1_N4)

nat_total = N2_0_rootlike * M02[0][0] + N2_1_rootlike * M02[0][1]
print(nat_total, np.log(nat_total))


# In log space
Q = np.array([[-0.1, 0.1],[0.1, -0.1]])
M02 = np.log(np.matmul(np.array([[1, 0]]), linalg.expm(Q*1)))
M13 = np.log(np.matmul(np.array([[0, 1]]), linalg.expm(Q*2)))

## If 2 is in deme 0
N2_0_M23 = np.log(np.matmul(np.array([[1, 0]]), linalg.expm(Q*1)))
N2_0_M24 = np.log(np.matmul(np.array([[1, 0]]), linalg.expm(Q*2)))
N2_0_M34 = np.array([logsumexp(np.log(linalg.expm(Q*1)).T + (M13 + N2_0_M23), axis=1)])
N2_0_N4 = N2_0_M24 + N2_0_M34
N2_0_rootlike = logsumexp(N2_0_N4)

## If 2 is in deme 1
N2_1_M23 = np.log(np.matmul(np.array([[0, 1]]), linalg.expm(Q*1)))
N2_1_M24 = np.log(np.matmul(np.array([[0, 1]]), linalg.expm(Q*2)))
N2_1_M34 = np.array([logsumexp(np.log(linalg.expm(Q*1)).T + (M13 + N2_1_M23), axis=1)])
N2_1_N4 = N2_1_M24 + N2_1_M34
N2_1_rootlike = logsumexp(N2_1_N4)

log_total = logsumexp([N2_0_rootlike + M02[0][0], N2_1_rootlike + M02[0][1]])
print(log_total)