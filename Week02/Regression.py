

import numpy as np
import matplotlib.pyplot as plt
import csv

X = []
Y = []

f = open('X.csv', 'r')
csvReader = csv.reader(f)

for row in csvReader:
    X.append(row)
    
f = open('Y.csv', 'r')
csvReader = csv.reader(f)

for row in csvReader:
    Y.append(row)
    
f.close()

X = np.asarray(X, dtype = 'float64')
Y = np.asarray(Y, dtype = 'float64')

xTemp = X[:, 0:2]

theta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(xTemp), xTemp)), np.transpose(xTemp)), Y)

Y_est = np.dot(xTemp, theta)

xTemp = X[:, 0:2]

theta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(xTemp), xTemp)), np.transpose(xTemp)), Y)

Y_est = np.dot(xTemp, theta)

m0, c0 = np.linalg.lstsq(xTemp, Y)[0]
m1, c1 = np.linalg.lstsq(xTemp, Y_est)[0]

newX = np.zeros((X.shape[0], 9))
newX[:, 0:2] = X[:, 0:2]
for i in range(2, 9):
    newX[:, i] = newX[:, 1] * newX[:, i-1]

newTheta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(newX), newX)), np.transpose(newX)), Y)

newY_est = np.dot(newX, newTheta)

m2, c2 = np.linalg.lstsq(xTemp, newY_est)[0]

plt.figure(1, figsize = (17, 5))

ax1 = plt.subplot(1, 3, 1)
plt.plot(X[:, 1], Y, 'ro', markeredgecolor = 'none')
plt.plot(X[:, 1], m0+c0*X[:, 1], 'r-')
plt.plot(X[:, 1], Y_est, 'bo', markeredgecolor = 'none')
plt.plot(X[:, 1], m1+c1*X[:, 1], 'b-')

ax2 = plt.subplot(1, 3, 2, sharey = ax1)
plt.plot(X[:, 1], Y, 'ro', markeredgecolor = 'none')
plt.plot(X[:, 1], newY_est, 'go', markeredgecolor = 'none')
plt.plot(X[:, 1], m2 + c2*X[:, 1], 'g-')

ax3 = plt.subplot(1, 3, 3, sharey = ax2)
plt.plot(X[:, 1], Y, 'ro', markeredgecolor = 'none')
plt.plot(X[:, 1], Y_est, 'bo', markeredgecolor = 'none')
plt.plot(X[:, 1], newY_est, 'go', markeredgecolor = 'none')

plt.show()