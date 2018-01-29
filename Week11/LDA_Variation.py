import numpy as np
from numpy import ndarray, sum
from scipy.special import polygamma, gamma as gammaFunc
from math import log, e, exp
import csv

class LDA_Variational:

    intNumTopic = 0
    numIterations = 0
    numInternalIterations = 100
    numNewtonIteration = 10

    intNumDoc = 0
    intUniqueWord = 0
    word = []
    numWordPerDoc = []
    corpusMatrix = []
    corpusList = []

    alpha = []
    beta = []

    fileLog = 0

    def __init__(self,strBagOfWordFileName,strUniqueWordFilename,strLogFilename,intNumTopic,numIterations):
        self.readBagOfWord(strBagOfWordFileName)
        self.readUniqueWord(strUniqueWordFilename)
        self.countNumWordinDocument()
        self.numIterations = numIterations
        self.numInternalIterations = numIterations
        self.intNumDoc = len(self.corpusMatrix)
        self.intUniqueWord = len(self.corpusMatrix[0])
        self.intNumTopic = intNumTopic
        self.fileLog = open(strLogFilename,'w')

        self.alpha = ndarray(shape=(self.intNumTopic),dtype=float)
        self.beta = ndarray(shape=(self.intNumTopic,self.intUniqueWord),dtype=float)
        for k in range(intNumTopic):
            self.alpha[k] = 1.0 / float(self.intNumTopic)

        for k in range(self.intNumTopic):
            for v in range(self.intUniqueWord):
                self.beta[k][v] = 1.0 / float(self.intUniqueWord) + np.random.rand(1)*0.01
            normalizeConstantBeta = sum(self.beta[k])
            for v in range(self.intUniqueWord):
                self.beta[k][v] = self.beta[k][v] / normalizeConstantBeta

    def __del__(self):
        self.logOut('')
        self.logOut('')
        self.logOut('------------------------------------------')
        self.logOut('FINAL RESULT')
        self.logLDA()
        self.fileLog.close()

    def logOut(self,msg):
        self.fileLog.write(msg)
        self.fileLog.write('\n')
        self.fileLog.flush()

    def readBagOfWord(self,strFileName):
        file = open(strFileName,'r')
        try:
            reader = csv.reader(file)
            for row in reader:
                rowInCorpus = []
                listInCorpus = []
                idx = 0
                for col in row:
                    rowInCorpus.append(int(col))
                    for itr in range(int(col)):
                        listInCorpus.append(idx)
                    idx = idx + 1
                self.corpusMatrix.append(rowInCorpus)
                self.corpusList.append(listInCorpus)
        finally:
            file.close()

    def readUniqueWord(self,strFileName):
        file = open(strFileName, 'r')
        try:
            while True:
                line = file.readline().strip()
                if not line:
                    break
                self.word.append(line)
        finally:
            file.close()

    def countNumWordinDocument(self,):
        for i in range(len(self.corpusMatrix)):
            wordInDoc = self.corpusMatrix[i]
            cnt = 0
            for j in range(len(wordInDoc)):
                cnt = cnt + wordInDoc[j]
            self.numWordPerDoc.append(cnt)

    def logLDA(self,topN=10):
        if self.intUniqueWord < 10:
            topN = self.intUniqueWord

        self.logOut('------------------------------------------')
        self.logOut('Alpha')
        self.logOut(str(self.alpha))
        self.logOut('')
        self.logOut('Beta')
        for k in range(self.intNumTopic):
            self.logOut(str(self.beta[k]))
        self.logOut('')
        self.logOut( 'Topic Words')
        for k in range(self.intNumTopic):
            order = []
            value = []
            for v in range(self.intUniqueWord):
                order.append(v)
                value.append(self.beta[k][v])
            for itr1 in range(self.intUniqueWord):
                for itr2 in range(itr1,self.intUniqueWord):
                    if value[itr1] < value[itr2]:
                        tempValue = value[itr1]
                        value[itr1] = value[itr2]
                        value[itr2] = tempValue

                        tempOrder = order[itr1]
                        order[itr1] = order[itr2]
                        order[itr2] = tempOrder
            self.logOut( 'Topic '+str(k) )
            for itr1 in range(topN):
                self.logOut( (self.word[order[itr1]]+'('+str(value[itr1])+')') )
            self.logOut('')
        self.logOut( 'Document Prob')
        theta = []
        for d in range(self.intNumDoc):
            prob = ndarray(shape=(self.intNumTopic),dtype=float)
            for n in range(self.numWordPerDoc[d]):
                idxWord = self.corpusList[d][n]
                for k in range(self.intNumTopic):
                    prob[k] = prob[k] + self.beta[k][idxWord]
            normalizeConstant = sum(prob)
            for k in range(self.intNumTopic):
                prob[k] = prob[k] / normalizeConstant
            self.logOut( str(prob) )
            theta.append(prob)

        self.logOut('')
        self.logOut( 'Topic Prob')
        topicProportion = ndarray(shape=(self.intNumTopic),dtype=float)
        for k in range(self.intNumTopic):
            for d in range(self.intNumDoc):
                topicProportion[k] = topicProportion[k] + theta[d][k]
        normalizeConstant = sum(topicProportion)
        for k in range(self.intNumTopic):
            topicProportion[k] = topicProportion[k] / normalizeConstant
        self.logOut( str(topicProportion) )
        self.logOut('------------------------------------------')

    def printLDA(self,topN):
        print 'Alpha'
        print self.alpha
        print 'Beta'
        for k in range(self.intNumTopic):
            print self.beta[k]
        print 'Topic Words'
        for k in range(self.intNumTopic):
            order = []
            value = []
            for v in range(self.intUniqueWord):
                order.append(v)
                value.append(self.beta[k][v])
            for itr1 in range(self.intUniqueWord):
                for itr2 in range(itr1,self.intUniqueWord):
                    if value[itr1] < value[itr2]:
                        tempValue = value[itr1]
                        value[itr1] = value[itr2]
                        value[itr2] = tempValue

                        tempOrder = order[itr1]
                        order[itr1] = order[itr2]
                        order[itr2] = tempOrder
            print 'Topic ',k
            for itr1 in range(topN):
                print self.word[order[itr1]],'(',value[itr1],'),',
            print
        print 'Document Prob'
        theta = []
        for d in range(self.intNumDoc):
            prob = ndarray(shape=(self.intNumTopic),dtype=float)
            for n in range(self.numWordPerDoc[d]):
                idxWord = self.corpusList[d][n]
                for k in range(self.intNumTopic):
                    prob[k] = prob[k] + self.beta[k][idxWord]
            normalizeConstant = sum(prob)
            for k in range(self.intNumTopic):
                prob[k] = prob[k] / normalizeConstant
            print prob
            theta.append(prob)

        print 'Topic Prob'
        topicProportion = ndarray(shape=(self.intNumTopic),dtype=float)
        for k in range(self.intNumTopic):
            for d in range(self.intNumDoc):
                topicProportion[k] = topicProportion[k] + theta[d][k]
        normalizeConstant = sum(topicProportion)
        for k in range(self.intNumTopic):
            topicProportion[k] = topicProportion[k] / normalizeConstant
        print topicProportion

    def performLDA(self):
        for iteration in range(self.numIterations):
            self.logLDA()
            self.logOut( 'Variational Iteration : '+str(iteration) )
            # E-Step : Learning phi and gamma

            # initialize the variational parameter
            phi = []
            gamma = ndarray(shape=(self.intNumDoc, self.intNumTopic), dtype=float)
            for d in range(self.intNumDoc):
                phi.append(ndarray(shape=(self.numWordPerDoc[d], self.intNumTopic), dtype=float))
                for n in range(self.numWordPerDoc[d]):
                    for k in range(self.intNumTopic):
                        phi[d][n][k] = 1.0 / float(self.intNumTopic)
            for k in range(self.intNumTopic):
                for d in range(self.intNumDoc):
                    gamma[d][k] = self.alpha[k] + float(self.intUniqueWord) / float(self.intNumTopic)

            for d in range(self.intNumDoc):
                for iterationInternal in range(self.numInternalIterations):
                    # Learning phi
                    for n in range(self.numWordPerDoc[d]):
                        for k in range(self.intNumTopic):
                            phi[d][n][k] = self.beta[k][self.corpusList[d][n]]*exp(polygamma(0,gamma[d][k]))
                        normalizeConstantPhi = sum(phi[d][n])
                        for k in range(self.intNumTopic):
                            phi[d][n][k] = phi[d][n][k] / normalizeConstantPhi
                    # Learning gamma
                    for k in range(self.intNumTopic):
                        gamma[d][k] = self.alpha[k]
                        for n in range(self.numWordPerDoc[d]):
                            gamma[d][k] = gamma[d][k] + phi[d][n][k]

            # M-Step : Learning alpha and beta

            # Learning Beta
            for k in range(self.intNumTopic):
                for d in range(self.intNumDoc):
                    for n in range(self.numWordPerDoc[d]):
                        self.beta[k][self.corpusList[d][n]] = self.beta[k][self.corpusList[d][n]] + phi[d][n][k]
                normalizeConstantBeta = sum(self.beta[k])
                for v in range(self.intUniqueWord):
                    self.beta[k][v] = self.beta[k][v] / normalizeConstantBeta

            # Learning Alpha
            # calculate current ELBO with respect to Current Alpha
            ELBOMax = 0
            for d in range(self.intNumDoc):
                ELBOMax = ELBOMax + log(gammaFunc(sum(self.alpha)), e)
                for k in range(self.intNumTopic):
                    ELBOMax = ELBOMax - log(gammaFunc(self.alpha[k]), e)
                    ELBOMax = ELBOMax + (self.alpha[k] - 1) * (
                    polygamma(0, gamma[d][k]) - polygamma(0, sum(gamma[d])))
            tempAlpha = ndarray(shape=(self.intNumTopic), dtype=float)
            bestAlpha = ndarray(shape=(self.intNumTopic), dtype=float)
            for k in range(self.intNumTopic):
                bestAlpha[k] = self.alpha[k]
                tempAlpha[k] = self.alpha[k]
            self.logOut( 'Newton-Rhapson Itr - ELBO : '+str(ELBOMax)+' | alpha : '+str(tempAlpha) )
            # Newton-Rhapson optimization
            for itr in range(self.numNewtonIteration):
                # Building Hessian Matrix and Derivative Vector
                H = ndarray(shape=(self.intNumTopic,self.intNumTopic), dtype=float)
                g = ndarray(shape=(self.intNumTopic), dtype=float)
                for k1 in range(self.intNumTopic):
                    g[k1] = float(self.intNumDoc)*(polygamma(0,sum(tempAlpha))-polygamma(0,tempAlpha[k1]))
                    for d in range(self.intNumDoc):
                        g[k1] = g[k1] + ( polygamma(0,gamma[d][k1]) - polygamma(0,sum(gamma[d])) )
                    for k2 in range(self.intNumTopic):
                        H[k1][k2] = 0
                        if k1 == k2:
                            H[k1][k2] = H[k1][k2] - float(self.intNumDoc) * polygamma(1,tempAlpha[k1])
                        H[k1][k2] = H[k1][k2] + float(self.intNumDoc) * polygamma(1,sum(tempAlpha))

                # Update Alpha in Log Domain
                deltaAlpha = np.dot(np.linalg.inv(H),g)

                for k in range(self.intNumTopic):
                    logAlphaK = log(tempAlpha[k],e)
                    logAlphaK = logAlphaK - deltaAlpha[k]
                    tempAlpha[k] = exp(logAlphaK)
                    if tempAlpha[k] < 0.00001:
                        tempAlpha[k] = 0.00001

                # calculate current ELBO with respect to New Alpha
                ELBOAfter = 0
                for d in range(self.intNumDoc):
                    ELBOAfter = ELBOAfter + log(gammaFunc(sum(tempAlpha)), e)
                    for k in range(self.intNumTopic):
                        ELBOAfter = ELBOAfter - log(gammaFunc(tempAlpha[k]),e)
                        ELBOAfter = ELBOAfter + (tempAlpha[k]-1)*(polygamma(0,gamma[d][k])-polygamma(0,sum(gamma[d])))

                self.logOut( 'Newton-Rhapson Itr - ELBO : '+str(ELBOAfter)+' | alpha : '+str(tempAlpha) )

                if ELBOMax <= ELBOAfter:
                    ELBOMax = ELBOAfter
                    for k in range(self.intNumTopic):
                        bestAlpha[k] = tempAlpha[k]

            self.alpha = bestAlpha

if __name__ == "__main__":
    #LDA = LDA_Variational('Sample-bow.csv','Sample-words.csv','log.txt',2,100)
    LDA = LDA_Variational('SentimentDataset-bow.csv','SentimentDataset-words.csv','log.txt',5,30)
    LDA.performLDA()
    LDA.printLDA(5)

