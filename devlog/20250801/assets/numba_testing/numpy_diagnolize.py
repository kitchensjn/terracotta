import numpy as np
from scipy import linalg


A = np.array([[-3,1,0,2,0],[1,-3,2,0,0],[0,2,-7,3,2],[2,0,3,-7,2],[0,0,2,2,-4]])

print(linalg.expm(A))

eigen = np.linalg.eig(A)

e_values = eigen.eigenvalues
e_values_exp = np.exp(e_values)
eD = np.diag(e_values_exp)

S = eigen.eigenvectors
Sinv = np.linalg.inv(S)

print(np.matmul(np.matmul(S, eD), Sinv))
