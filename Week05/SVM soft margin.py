import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

pos = np.array([[3, 7], [4, 6], [5, 6], [7, 7], [8, 5], [5, 5.2], [7, 5]])
neg = np.array([[4, 5], [5, 5], [6, 3], [7, 4], [9, 4]])

X = np.ones((pos.shape[0]+neg.shape[0], 2))
X[0:pos.shape[0], :] = pos
X[pos.shape[0]:pos.shape[0]+neg.shape[0], :] = neg

Y = np.ones(pos.shape[0] + neg.shape[0])
Y[0:pos.shape[0]] = 1
Y[pos.shape[0]:pos.shape[0]+neg.shape[0]] = -1

plt.figure(1, figsize = (7, 7))
plt.plot(X[0:pos.shape[0], 0], X[0:pos.shape[0], 1], 'bo', label = 'positive')
plt.plot(X[pos.shape[0]:pos.shape[0] + neg.shape[0], 0], X[pos.shape[0]:pos.shape[0] + neg.shape[0], 1], 'ro', label = 'negative')
plt.legend()
plt.show()

A = np.zeros((X.shape[0], X.shape[1] + 1))
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        A[i, j] = X[i, j] * Y[i] * (-1)
    A[i, -1] = Y[i] * (-1)

b = np.ones(X.shape[0]) * (-1)

H = np.zeros((X.shape[1] + 1, X.shape[1] + 1))
for i in range(X.shape[1]):
    H[i, i] = 1

f = np.zeros(X.shape[1] + 1)

H = matrix(H)
f = matrix(f)
A = matrix(A)
b = matrix(b)

sol = solvers.qp(H, f, A, b)
w = sol['x']

plt.figure(1, figsize = (7, 7))
plt.plot(X[0:pos.shape[0], 0], X[0:pos.shape[0], 1], 'bo', label = 'positive')
plt.plot(X[pos.shape[0]:pos.shape[0] + neg.shape[0], 0], X[pos.shape[0]:pos.shape[0] + neg.shape[0], 1], 'ro', label = 'negative')
plt.legend()

lineX = [1, 10]

lineY0 = [(w[2] + w[0]*lineX[0])/-w[1], (w[2] + w[0]*lineX[1])/-w[1]]
plt.plot(lineX, lineY0, 'm-')
plt.show()

plt.figure(1, figsize=(15, 23))
C = [0.1, 0.3, 0.5, 1, 10, 10000]
for ite in range(len(C)):
    print('%s. C = %s\n' % (ite + 1, C[ite]))
    A = np.zeros((X.shape[0] + X.shape[0], X.shape[1] + X.shape[0] + 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            A[i, j] = X[i, j] * Y[i]
        A[i, X.shape[1]] = Y[i]
        A[i, X.shape[1] + i + 1] = -1

    for i in range(X.shape[0]):
        A[i + X.shape[0], X.shape[1] + i + 1] = -1

    b = np.zeros((X.shape[0] + X.shape[0], 1))
    b[0:X.shape[0]] = -1

    H = np.zeros((X.shape[1] + 1 + X.shape[0], X.shape[1] + 1 + X.shape[0]))
    for i in range(X.shape[1]):
        H[i, i] = 1

    f = np.zeros(X.shape[1] + 1 + X.shape[0])
    for i in range(X.shape[1] + 1, X.shape[1] + 1 + X.shape[0]):
        f[i] = C[ite]

    H = matrix(H)
    f = matrix(f)
    A = matrix(A)
    b = matrix(b)

    sol = solvers.qp(H, f, A, b)
    w = sol['x']

    plt.subplot(3, 2, ite + 1)
    plt.title('C = %s\n' % C[ite], fontsize=20)
    plt.axis([1, 10, 3, 7])
    plt.plot(X[0:pos.shape[0], 0], X[0:pos.shape[0], 1], 'bo', label='positive')
    plt.plot(X[pos.shape[0]:pos.shape[0] + neg.shape[0], 0], X[pos.shape[0]:pos.shape[0] + neg.shape[0], 1], 'ro',
             label='negative')
    plt.legend()
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], '%.1e' % (w[i + 3]))
    print('')
plt.show()

