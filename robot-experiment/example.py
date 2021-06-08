import numpy as np
import matplotlib.pyplot as plt

length = 10
A = np.zeros((length+2, length))
for idx in range(length):
    A[idx, idx] = 1 * (idx+1)
    A[idx+1,idx] = -2 * (idx+1)
    A[idx+2,idx] = 1 * (idx+1)
print(A.T @ A)
R = np.linalg.inv(A.T @ A)

plt.plot(R)
plt.show()
