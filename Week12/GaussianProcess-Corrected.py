# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

# In[2]:

def KernelHyperParameterLearning(trainingX, trainingY):
    numDataPoints = len(trainingY)
    numDimension = len(trainingX[0])

    # Input and Output Data Declaration for TensorFlow
    obsX = tf.placeholder(tf.float32, [numDataPoints, numDimension])
    obsY = tf.placeholder(tf.float32, [numDataPoints, 1])

    # Learning Parameter Variable Declaration for TensorFlow
    theta0 = tf.Variable(1.0)
    theta1 = tf.Variable(1.0)
    theta2 = tf.Variable(1.0)
    theta3 = tf.Variable(1.0)
    beta = tf.Variable(1.0)

    # Kernel building
    matCovarianceLinear = []
    for i in range(numDataPoints):
        for j in range(numDataPoints):
            kernelEvaluationResult = KernelFunctionWithTensorFlow(theta0, theta1, theta2, theta3,
                                                                  tf.slice(obsX, [i, 0], [1, numDimension]),
                                                                  tf.slice(obsX, [j, 0], [1, numDimension]))
            if i != j:
                matCovarianceLinear.append(kernelEvaluationResult)
            if i == j:
                matCovarianceLinear.append(kernelEvaluationResult + tf.div(1.0, beta))
    matCovarianceCombined = tf.stack(matCovarianceLinear)
    matCovariance = tf.reshape(matCovarianceCombined, [numDataPoints, numDataPoints])
    matCovarianceInv = tf.reciprocal(matCovariance)

    # Prediction for calculating sum of sqaured error
    sumsquarederror = 0.0
    for i in range(numDataPoints):
        k = tf.Variable(tf.ones([numDataPoints]))
        for j in range(numDataPoints):
            kernelEvaluationResult = KernelFunctionWithTensorFlow(theta0, theta1, theta2, theta3,
                                                                  tf.slice(obsX, [i, 0], [1, numDimension]),
                                                                  tf.slice(obsX, [j, 0], [1, numDimension]))
            indices = tf.constant([j])
            tempTensor = tf.Variable(tf.zeros([1]))
            tempTensor = tf.add(tempTensor, kernelEvaluationResult)
            tf.scatter_update(k, tf.reshape(indices, [1, 1]), tempTensor)

        c = tf.Variable(tf.zeros([1, 1]))
        kernelEvaluationResult = KernelFunctionWithTensorFlow(theta0, theta1, theta2, theta3,
                                                              tf.slice(obsX, [i, 0], [1, numDimension]),
                                                              tf.slice(obsX, [i, 0], [1, numDimension]))
        c = tf.div(tf.add(tf.add(c, kernelEvaluationResult), 1), beta)

        k = tf.reshape(k, [1, numDataPoints])

        predictionMu = tf.matmul(k, tf.matmul(matCovarianceInv, obsY))
        predictionVar = tf.subtract(c, tf.matmul(k, tf.matmul(matCovarianceInv, tf.transpose(k))))

        sumsquarederror = tf.add(sumsquarederror, tf.pow(tf.subtract(predictionMu, tf.slice(obsY, [i, 0], [1, 1])), 2))

    # Training session declaration
    training = tf.train.GradientDescentOptimizer(0.0001).minimize(sumsquarederror)

    # Session initialization
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init = tf.initialize_all_variables()
    sess.run(init)

    # Session Running for Training
    for i in range(100):
        sess.run(training, feed_dict={obsX: trainingX, obsY: trainingY})

        trainedTheta = []
        trainedTheta.append(sess.run(theta0, feed_dict={obsX: trainingX, obsY: trainingY}))
        trainedTheta.append(sess.run(theta1, feed_dict={obsX: trainingX, obsY: trainingY}))
        trainedTheta.append(sess.run(theta2, feed_dict={obsX: trainingX, obsY: trainingY}))
        trainedTheta.append(sess.run(theta3, feed_dict={obsX: trainingX, obsY: trainingY}))
        trainedBeta = sess.run(beta, feed_dict={obsX: trainingX, obsY: trainingY})
        # print "---------------------- Iteration ",i," -----------------"
        # print "Sum of Squared Error : ",sess.run(sumsquarederror, feed_dict={obsX: trainingX, obsY: trainingY})
        # print "Theta : ",trainedTheta
        # print "Beta : ",trainedBeta

    # Return Learning Result
    return trainedTheta, trainedBeta


# In[5]:

def KernelFunctionWithTensorFlow(theta0, theta1, theta2, theta3, X1, X2):
    insideexp1 = tf.multiply(tf.div(theta1, 2.0), np.dot((X1 - X2), (X1 - X2)))
    insideexp2 = theta2
    
    insideexp3 = tf.multiply(theta3, np.dot(np.transpose(X1), X2))
    #insideexp = tf.add(tf.add(insideexp1, insideexp2), insideexp3)
    ret = theta0*tf.exp(-insideexp1)+insideexp2+insideexp3
    #ret = tf.multiply(theta0, tf.exp(insideexp))
    return ret


# In[6]:

def KernelFunctionWihtoutTensorFlow(theta, X1, X2):
    ret = theta[0] * np.exp(np.multiply(-theta[1] / 2, np.dot(np.subtract(X1, X2), np.subtract(X1, X2)))) + theta[2] + \
          theta[3] * np.dot(np.transpose(X1), X2)
    return ret


# In[7]:

def PredictionGaussianProcessRegression(theta, beta, C_inv, numPoints, sampleX, sampleY, inputElement):
    k = np.zeros(numPoints)
    for i in range(numPoints):
        k[i] = KernelFunctionWihtoutTensorFlow(theta, sampleX[i], inputElement)

    c = KernelFunctionWihtoutTensorFlow(theta, inputElement, inputElement) + 1.0 / beta
    #print(np.shape(k))
    mu = np.dot(k, np.dot(C_inv, sampleY))
    var = c - np.dot(k, np.dot(C_inv, k))

    return mu[0], var


# In[8]:

def KernelCalculation(theta, beta, numPoints, sampleX):
    C = np.zeros((numPoints, numPoints))

    for i in range(numPoints):
        for j in range(numPoints):
            C[i, j] = KernelFunctionWihtoutTensorFlow(theta, sampleX[i], sampleX[j])
            if i == j:
                C[i, j] += 1.0 / beta

    return C, np.linalg.inv(C)


# In[9]:

def PlottingGaussianProcessRegression(plotN, strTitle, inputs, mu_next, sigma2_next, sampleX, sampleY):
    plt.subplot(5, 2, plotN)
    plt.xlim([0, 6])
    plt.ylim([-3, 3])
    plt.title(strTitle)

    plt.fill_between(inputs, mu_next - 2 * np.sqrt(sigma2_next), mu_next + 2 * np.sqrt(sigma2_next),
                     color=(0.9, 0.9, 0.9))
    plt.plot(sampleX, sampleY, 'r+', markersize=10)
    plt.plot(X, trueY, 'g-')
    plt.plot(inputs, mu_next, 'bo-', markersize=5)


# In[10]:

'''snr = 0.2
numObservePoints = 5
numInputDimension = 1

X = np.arange(0, 2 * np.pi, 0.1)
numTruePoints = X.shape[0]
trueY = np.sin(X)

# In[11]:

trainingX = []
trainingY = []

for itr2 in range(numObservePoints):
    sampleX = 2 * np.pi * np.random.random()
    trainingX.append([sampleX])
    trainingY.append([np.sin(sampleX) + snr * np.random.randn()])
    numPoints = len(trainingX)

print(trainingX)
print(trainingY)

# In[12]:

trainedTheta, trainedBeta = KernelHyperParameterLearning(trainingX, trainingY)
print("---------------------- Trained Result -----------------")
print("Theta : ", trainedTheta)
print("Beta : ", trainedBeta)

# # Gausssian Process Regression with Kernel learning

# In[13]:

sampleXs = []
sampleYs = []

showVisualization = [1, 5, 10, 14, 19]
numMaxPoints = 20
theta = np.array([1, 1, 1, 1])
beta = 300
plt.figure(1, figsize=(14, 25), dpi=100)
plotN = 1

print("iteration : ",)
for itr2 in range(numMaxPoints):
    print(itr2, " ",)
    sampleX = 2 * np.pi * np.random.random()
    sampleXs.append([sampleX])
    sampleYs.append([np.sin(sampleX) + snr * np.random.randn()])

    inputs = np.arange(0, 2 * np.pi, 0.3)

    mu_next = []
    sigma2_next = []

    if itr2 in showVisualization:
        C, C_inv = KernelCalculation(theta, beta, len(sampleYs), sampleXs)

        for itr1 in range(len(inputs)):
            mu, var = PredictionGaussianProcessRegression(theta, beta, C_inv, len(sampleYs), sampleXs, sampleYs,
                                                          inputs[itr1])
            mu_next.append(mu)
            sigma2_next.append(var)

        PlottingGaussianProcessRegression(plotN,
                                          'Without Kernel Parameter Learning After %s sampling\n' % (len(sampleYs)),
                                          inputs, mu_next, sigma2_next, sampleXs, sampleYs)
        plotN += 1

    mu_next = []
    sigma2_next = []
    if itr2 in showVisualization:
        trainedTheta, trainedBeta = KernelHyperParameterLearning(sampleXs, sampleYs)
        C, C_inv = KernelCalculation(trainedTheta, trainedBeta, len(sampleYs), sampleXs)

        for itr1 in range(len(inputs)):
            mu, var = PredictionGaussianProcessRegression(trainedTheta, trainedBeta, C_inv, len(sampleYs), sampleXs,
                                                          sampleYs, inputs[itr1])
            mu_next.append(mu)
            sigma2_next.append(var)

        PlottingGaussianProcessRegression(plotN, 'With Kernel Parameter Learning After %s sampling\n' % (len(sampleYs)),
                                          inputs, mu_next, sigma2_next, sampleXs, sampleYs)
        plotN += 1
plt.show()'''

# # Bayeisan Optimization with Kernel Learning

# In[14]:

def AcquisitionFunctionProbImprovement(sampleX, sampleY, m, Xs, Mus, Sigmas):
    numSamples = len(sampleY)

    idxMax = -1
    Ymax = -1000000
    for itr in range(numSamples):
        if sampleY[itr][0] > Ymax:
            Ymax = sampleY[itr][0]
            idxMax = itr

    probImprovements = []
    for itr in range(len(Mus)):
        probImprovements.append((Mus[itr] - (1 + m) * Ymax) / np.sqrt(Sigmas[itr]))
        probImprovements[itr] = norm.cdf(probImprovements[itr])

    print("Prob Improvements : ",probImprovements)

    idxProbMax = 0
    Probmax = -1
    for itr in range(len(Mus)):
        if probImprovements[itr] > Probmax:
            Probmax = probImprovements[itr]
            idxProbMax = itr

    return Xs[idxProbMax]


# In[15]:

def AcquisitionFunctionExpectedImprovement(sampleX, sampleY, Xs, Mus, Sigmas):
    numSamples = len(sampleY)

    idxMax = -1
    Ymax = -1000000
    for itr in range(numSamples):
        if sampleY[itr][0] > Ymax:
            Ymax = sampleY[itr][0]
            idxMax = itr

    expectedImprovements = []
    for itr in range(len(Mus)):
        u = (Ymax - Mus[itr]) / np.sqrt(Sigmas[itr])
        expectedImprovements.append(np.sqrt(Sigmas[itr]) * (-u * norm.cdf(-u) + norm.pdf(u)))

    print("Expected Improvements : ",expectedImprovements)

    idxEIMax = 0
    EImax = -1
    for itr in range(len(Mus)):
        if expectedImprovements[itr] > EImax:
            EImax = expectedImprovements[itr]
            idxEIMax = itr

    return Xs[idxEIMax]

def AcquisitionFunctionNewExpectedImprovement(sampleX, sampleY, Xs, Mus, Sigmas):
    numSamples = len(sampleY)

    idxMax = -1
    Ymax = -1000000
    for itr in range(numSamples):
        if sampleY[itr][0] > Ymax:
            Ymax = sampleY[itr][0]
            idxMax = itr

    expectedImprovements = []
    for itr in range(len(Mus)):
        u = (Ymax - Mus[itr]) / np.sqrt(Sigmas[itr])
        expectedImprovements.append(Sigmas[itr] * (-u * norm.pdf(u) + (1 + pow(u,2)) * norm.cdf(-u)))

    print("Expected Improvements : ",expectedImprovements)

    idxEIMax = 0
    EImax = -1
    for itr in range(len(Mus)):
        if expectedImprovements[itr] > EImax:
            EImax = expectedImprovements[itr]
            idxEIMax = itr

    return Xs[idxEIMax]


# In[16]:

snr = 0.2
sampleXs = []
sampleYs = []

showVisualization = [1, 5, 10, 20, 49]
numTrials = 50
trainedTheta = np.array([1, 1, 1, 1])
trainedBeta = 1
m = 1
kernelLearning = False
# acquisitionFunction = 'ProbImprovement'
acquisitionFunction = 'NewExpectedImprovement'

plt.figure(1, figsize=(14, 25), dpi=100)
plotN = 1

for itr2 in range(1):
    sampleX = 2 * np.pi * np.random.random()
    sampleXs.append([sampleX])
    sampleYs.append([np.sin(sampleX) + snr * np.random.randn()])



for itr2 in range(numTrials):

    Xs = np.arange(0, 2 * np.pi, 0.01)
    Mus = []
    Sigmas = []
    print("Learning Kernel Parameters.........")
    if kernelLearning == True:
        trainedTheta, trainedBeta = KernelHyperParameterLearning(sampleXs, sampleYs)
    C, C_inv = KernelCalculation(trainedTheta, trainedBeta, len(sampleYs), sampleXs)
    print("Trained Kernel Parameters : ", trainedTheta, trainedBeta)
    print("Calculating Predicted Values.........")
    #print(np.shape(C_inv))
    #print(len(sampleYs))
    for itr1 in range(len(Xs)):
        mu, var = PredictionGaussianProcessRegression(trainedTheta, trainedBeta, C_inv, len(sampleYs), sampleXs,
                                                      sampleYs, Xs[itr1])
        Mus.append(mu)
        Sigmas.append(var)
    print("Calculated Results : ",Mus,Sigmas)

    print("Calculating Acquisition Values.........")
    if acquisitionFunction == 'ProbImprovement':
        nextX = AcquisitionFunctionProbImprovement(sampleXs, sampleYs, m, Xs, Mus, Sigmas)
    if acquisitionFunction == 'ExpectedImprovement':
        nextX = AcquisitionFunctionExpectedImprovement(sampleXs, sampleYs, Xs, Mus, Sigmas)
    if acquisitionFunction == 'NewExpectedImprovement':
        nextX = AcquisitionFunctionNewExpectedImprovement(sampleXs, sampleYs, Xs, Mus, Sigmas)

    print("Learning Kernel Parameters.........")
    sampleXs.append([nextX])
    sampleYs.append([np.sin(nextX) + snr * np.random.randn()])
    if kernelLearning == True:
        trainedTheta, trainedBeta = KernelHyperParameterLearning(sampleXs, sampleYs)
    #print("Hoho")
    if itr2 in showVisualization:
        C, C_inv = KernelCalculation(trainedTheta, trainedBeta, len(sampleYs), sampleXs)
        print("Trained Kernel Parameters : ", trainedTheta, trainedBeta)
    
        print("iteration, probing point : ",itr2," ",nextX)
        PlottingGaussianProcessRegression(plotN, 'With Kernel Parameter Learning After %s sampling\n' % (len(sampleYs)), Xs,
                                          Mus, Sigmas, sampleXs, sampleYs)
        plotN += 1
plt.show()

