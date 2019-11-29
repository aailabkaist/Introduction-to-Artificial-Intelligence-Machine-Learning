import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

pos = np.array([[3, 7], [4, 6], [5, 6], [7, 7], [8, 5], [5, 5.2], [7, 5], [6, 3.75], [6, 4], [6, 5], [7, 5], [6, 4.5], [7, 4.5]])
neg = np.array([[4, 5], [5, 5], [6, 3], [7, 4], [9, 4], [5, 4], [5, 4.5], [5, 3.5], [7, 3.5]])

X = np.ones((pos.shape[0] + neg.shape[0], 9))
X[0:pos.shape[0], 0:2] = pos
X[pos.shape[0]:pos.shape[0]+neg.shape[0], 0:2] = neg

Y = np.ones(pos.shape[0] + neg.shape[0])
Y[0:pos.shape[0]] = 1
Y[pos.shape[0]:pos.shape[0]+neg.shape[0]] = -1

X[:, 2] = X[:, 0] * X[:, 0]
X[:, 3] = X[:, 1] * X[:, 1]
X[:, 4] = X[:, 0] * X[:, 1]

X[:, 5] = X[:, 2] * X[:, 0]
X[:, 6] = X[:, 3] * X[:, 1]
X[:, 7] = X[:, 4] * X[:, 0]
X[:, 8] = X[:, 4] * X[:, 1]

C = 10

A = np.zeros((X.shape[0] + X.shape[0], X.shape[1] + X.shape[0] + 1))

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        A[i, j] = X[i, j] * Y[i]

    A[i, X.shape[1]] = Y[i]
    A[i, X.shape[1] + i + 1] = -1

for i in range(X.shape[0]):
    A[i + X.shape[0], X.shape[1] + i + 1] = -1

b = np.zeros(X.shape[0] + X.shape[0])
b[0:X.shape[0]] = -1

H = np.zeros((X.shape[1] + 1 + X.shape[0], X.shape[1] + 1 + X.shape[0]))
for i in range(X.shape[1]):
    H[i, i] = 1

f = np.zeros(X.shape[1] + 1 + X.shape[0])
for i in range(X.shape[1] + 1, X.shape[1] + 1 + X.shape[0]):
    f[i] = C

H = matrix(H)
f = matrix(f)
A = matrix(A)
b = matrix(b)

sol = solvers.qp(H, f, A, b)
w = np.asarray(sol['x'])
a = -np.dot(A, w)

plt.figure(figsize=(6, 6))
plt.axis((1, 10, 3, 7))
plt.figure(1, figsize=(7, 7))
plt.plot(X[0:pos.shape[0], 0], X[0:pos.shape[0], 1], 'bo', label='positive')
plt.plot(X[pos.shape[0]:pos.shape[0] + neg.shape[0], 0], X[pos.shape[0]:pos.shape[0] + neg.shape[0], 1], 'ro',
         label='negative')
plt.legend()
for i in range(X.shape[0]):
    plt.text(X[i, 0], X[i, 1], '%.2e\n' % (w[i + int(X.shape[1]) + 1]))

lineX = [1, 10]

lineY0 = [(w[2] + w[0] * lineX[0]) / -w[1], (w[2] + w[0] * lineX[1]) / -w[1]]
plt.plot(lineX, lineY0, 'c')

margin = 1 / np.linalg.norm(w[0:2])

lineY1 = [(w[2] + w[0] * lineX[0]) / -w[1] + margin, (w[2] + w[0] * lineX[1]) / -w[1] + margin]
plt.plot(lineX, lineY1, 'b-')

lineY2 = [(w[2] + w[0] * lineX[0]) / -w[1] - margin, (w[2] + w[0] * lineX[1]) / -w[1] - margin]
plt.plot(lineX, lineY2, 'r-')

a = -1 * np.dot(A, w)

plt.show()

x1 = np.linspace(1, 10)
x2 = np.linspace(3, 7)
xx1, xx2 = np.meshgrid(x1, x2)

f = np.zeros((len(x1), len(x2)))

for i in range(len(x1)):
    for j in range(len(x2)):
        temp = np.zeros((X.shape[1], 1))
        temp[0] = x1[i]
        temp[1] = x2[j]
        temp[2] = x1[i]*x1[i]
        temp[3] = x2[j]*x2[j]
        temp[4] = x1[i]*x2[j]
        temp[5] = x1[i]*x1[i]*x1[i]
        temp[6] = x2[j]*x2[j]*x2[j]
        temp[7] = x1[i]*x1[i]*x2[j]
        temp[8] = x1[i]*x2[j]*x2[j]
        f[j, i] = np.dot(w[0:X.shape[1]].T, temp) + w[X.shape[1]]

lev = np.arange(-100, 100, 1)

fig = plt.figure(1, figsize = (10, 7))
plt.axis((1, 10, 3, 7))

plt.plot(X[0:pos.shape[0], 0], X[0:pos.shape[0], 1], 'bo', label = 'positive')
plt.plot(X[pos.shape[0]:pos.shape[0] + neg.shape[0], 0], X[pos.shape[0]:pos.shape[0] + neg.shape[0], 1], 'ro', label = 'negative')

plt.legend()

for i in range(X.shape[0]):
    plt.text(X[i, 0], X[i, 1], '%.2e\n'%(w[i + int(X.shape[1]) + 1]))

c = plt.contour(x1, x2, f, levels = lev, linestyles = 'solid')

cbar = plt.colorbar(c)

fig = plt.figure(figsize = (10, 10))

ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(xx1, xx2, f, rstride=5, cstride=5, cmap='RdBu_r', alpha = 0.3)
ax.scatter(X[0:pos.shape[0], 0], X[0:pos.shape[0], 1], np.dot(w[0:X.shape[1]].T, X[0:pos.shape[0], :].T) + w[X.shape[1]], c = 'b', s = 50)
ax.scatter(X[pos.shape[0]:pos.shape[0] + neg.shape[0], 0], X[pos.shape[0]:pos.shape[0] + neg.shape[0], 1],
           np.dot(w[0:X.shape[1]].T, X[pos.shape[0]:pos.shape[0] + neg.shape[0]].T) + w[X.shape[1]], c = 'r', s = 50)
ax.plot_wireframe(xx1, xx2, np.zeros((len(x1), len(x2))), rstride=5, cstride=5, colors = 'k', alpha = 0.5)
plt.contour(x1, x2, f, levels = [0], colors = 'k', linestyles = 'solid')
plt.show()