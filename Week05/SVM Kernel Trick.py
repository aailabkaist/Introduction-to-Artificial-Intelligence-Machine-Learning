import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


def Kernel(X1, X2):
    selection = 2
    param = 5
    param1 = 1000
    param2 = -100

    if selection == 1:
        return pow(np.dot(X1, X2), param)
    elif selection == 2:
        return pow(np.dot(X1, X2) + 1, param)
    elif selection == 3:
        return np.exp(-param * np.dot((X1 - X2), (X1 - X2)))
    elif selection == 4:
        return np.tanh(np.dot(X1, X2) * param1 + param2)

pos = np.array([[3, 7], [4, 6], [5, 6], [7, 7], [8, 5], [5, 5.2], [7, 5], [6, 3.75], [6, 4], [6, 5], [7, 5], [6, 4.5], [7, 4.5]])
neg = np.array([[4, 5], [5, 5], [6, 3], [7, 4], [9, 4], [5, 4], [5, 4.5], [5, 3.5], [7, 3.5]])

C = 1

X = np.ones((pos.shape[0]+neg.shape[0], 2))
X[0:pos.shape[0], :] = pos
X[pos.shape[0]:pos.shape[0]+neg.shape[0], :] = neg

Y = np.ones(pos.shape[0] + neg.shape[0])
Y[0:pos.shape[0]] = 1
Y[pos.shape[0]:pos.shape[0]+neg.shape[0]] = -1

H = np.zeros((pos.shape[0] + neg.shape[0], pos.shape[0] + neg.shape[0]))

for i in range(pos.shape[0] + neg.shape[0]):
    for j in range(pos.shape[0] + neg.shape[0]):
        H[i, j] += Kernel(X[i, :], X[j, :]) * Y[i] * Y[j] * -1

f = np.ones(pos.shape[0] + neg.shape[0])

H = -1 * H
f = -1 * f

A = np.zeros((pos.shape[0] + neg.shape[0] + pos.shape[0] + neg.shape[0], pos.shape[0] + neg.shape[0]))
b = np.zeros(pos.shape[0] + neg.shape[0] + pos.shape[0] + neg.shape[0])

for i in range(pos.shape[0] + neg.shape[0]):
    A[i, i] = -1
    A[i + pos.shape[0] + neg.shape[0], i] = 1
    b[i + pos.shape[0] + neg.shape[0]] = C

Aeq = np.zeros((1, pos.shape[0] + neg.shape[0]))
Beq = np.zeros((1, 1))

for i in range(pos.shape[0] + neg.shape[0]):
    Aeq[0, i] = Y[i]

f = matrix(f)
H = matrix(H)
A = matrix(A)
b = matrix(b)
Aeq = matrix(Aeq)
Beq = matrix(Beq)

sol = solvers.qp(H, f, A, b, Aeq, Beq)
alpha = sol['x']

bs = np.zeros(X.shape[0])
k = -1

for j in range(X.shape[0]):
    temp = 0
    for i in range(X.shape[0]):
        temp += alpha[i] * Y[i] * Kernel(X[i], X[j])
    bs[j] = Y[j] - temp

    if alpha[j] > 0.0001 and alpha[j] < C - 0.0001:
        k = j

b = bs[k]

a = np.zeros(X.shape[0])
for j in range(X.shape[0]):
    temp = 0
    for i in range(X.shape[0]):
        temp += alpha[i] * Y[i] * Kernel(X[i, :], X[j, :])
    a[j] = temp + b

x1 = np.linspace(1, 10)
x2 = np.linspace(3, 7)
xx1, xx2 = np.meshgrid(x1, x2)

f = np.zeros((len(x1), len(x2)))

for i in range(len(x1)):
    for j in range(len(x2)):
        temp = [x1[i], x2[j]]
        f[j, i] = 0
        for k in range(len(alpha)):
            f[j, i] += alpha[k] * Y[k] * Kernel(X[k, :], temp)
        f[j, i] += b

lev = np.arange(-10, 10, 5)
plt.figure(1, figsize = (7, 7))
plt.plot(X[0:pos.shape[0], 0], X[0:pos.shape[0], 1], 'b+', label = 'positive')
plt.plot(X[pos.shape[0]:pos.shape[0] + neg.shape[0], 0],
         X[pos.shape[0]:pos.shape[0] + neg.shape[0], 1], 'ro', markeredgecolor = 'None', label = 'negative')

c = plt.contour(x1, x2, f, levels = [-0.01], colors = 'g', linestyles = 'solid')
plt.legend()
for i in range(X.shape[0]):
    plt.text(X[i, 0], X[i, 1], '%.2f\n'%(a[i]))

fig = plt.figure(figsize = (10, 10))

ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.scatter(X[0:pos.shape[0], 0], X[0:pos.shape[0], 1], a[0:pos.shape[0]], c = 'b', s = 50)
ax.scatter(X[pos.shape[0]:pos.shape[0] + neg.shape[0], 0],
           X[pos.shape[0]:pos.shape[0] + neg.shape[0], 1],
           a[pos.shape[0]:pos.shape[0] + neg.shape[0]], c = 'r', s = 50)
ax.plot_surface(xx1, xx2, f, rstride=1, cstride=1, cmap='RdBu', alpha = 0.3)

ax.plot_wireframe(xx1, xx2, np.zeros((len(x1), len(x2))), rstride=5, cstride=5, color = 'k', alpha = 0.5)

plt.contour(x1, x2, f, levels = [0], colors = 'k', linestyles = 'solid')
plt.show()