import time
import os
import numpy as np
from numpy import ndarray, sum, zeros, ones
from scipy.special import polygamma, gammaln, digamma
from math import log, e, exp, sqrt
from scipy.stats import dirichlet as Dirichlet
from TONKPythonLib.Utility.BagOfWordReader import BagOfWordReader
from TONKPythonLib.Utility.BagOfWordFilter import BagOfWordFilter
from TONKPythonLib.Utility.SparseBagOfWordReader import SparseBagOfWordReader

class LDA_StochasticVariational:

    intNumTopic = 0
    numIterations = 0
    numInternalIterations = 10
    numNewtonIteration = 10

    intNumDoc = 0
    intUniqueWord = 0
    sizeMiniBatch = 0
    learningRate = 0
    discountInLearningRatePerItr = 0

    matrixDataset = []
    wordHeading = []
    idxWordHash = {}
    docHeading = []
    numWordPerDoc = []
    corpusMatrix = []
    corpusList = []

    alpha = []
    beta = []
    phi = []
    gamma = []

    fileLog = 0
    logEnabled = 0
    logCycle = 0

    def __init__(self,matrixDataset,intNumTopic,numIterations,sizeMiniBatch=2,learningRate=0.9,discountInLearningRatePerItr=1.0,strLogFilename='log.txt',logEnabled=False,logCycle=10,numLocalIterations=10):
        self.matrixDataset = matrixData
        self.corpusMatrix = self.matrixDataset.getMatrix()
        self.corpusList = self.createBagOfWordList(self.corpusMatrix)
        self.wordHeading = self.matrixDataset.getColHeading()
        for v in range(len(self.wordHeading)):
            self.idxWordHash[self.wordHeading[v]] = v
        self.docHeading = self.matrixDataset.getRowHeading()
        self.numWordPerDoc = self.countNumWordinDocument(self.corpusMatrix)
        self.numIterations = numIterations
        self.numInternalIterations = numLocalIterations
        self.intNumDoc = len(self.corpusMatrix)
        self.intUniqueWord = len(self.corpusMatrix[0])
        self.intNumTopic = intNumTopic
        self.sizeMiniBatch = sizeMiniBatch
        self.learningRate = learningRate
        self.discountInLearningRatePerItr = discountInLearningRatePerItr

        self.logCycle = logCycle
        self.logEnabled = logEnabled
        if self.logEnabled == True:
            idxLastSlash = strLogFilename.rfind("/")
            if os.path.exists(strLogFilename[0:idxLastSlash]) == False:
                os.makedirs(strLogFilename[0:idxLastSlash])
            self.fileLog = open(strLogFilename, 'w')

        self.alpha = zeros(shape=(self.intNumTopic),dtype=float)
        self.beta = zeros(shape=(self.intNumTopic,self.intUniqueWord),dtype=float)
        for k in range(intNumTopic):
            self.alpha[k] = 1.0 / float(self.intNumTopic)

        for k in range(self.intNumTopic):
            for v in range(self.intUniqueWord):
                self.beta[k][v] = 1.0 / float(self.intUniqueWord) + np.random.rand(1)*0.01
            normalizeConstantBeta = sum(self.beta[k])
            for v in range(self.intUniqueWord):
                self.beta[k][v] = self.beta[k][v] / normalizeConstantBeta

    def __del__(self):
        if self.logEnabled == True:
            self.logOut('')
            self.logOut('')
            self.logOut('------------------------------------------')
            self.logOut('FINAL RESULT')
            self.logOut("Log Likelihood : " + str(self.calculateLogLikelihoodBlei()))
            self.logLDA()
            self.fileLog.close()

    def createBagOfWordList(self,corpusMatrix):
        ret = []
        for i in range(len(corpusMatrix)):
            listInCorpus = []
            for j in range(len(corpusMatrix[i])):
                for itr in range(corpusMatrix[i][j]):
                    listInCorpus.append(j)
            ret.append(listInCorpus)
        return ret

    def logOut(self,msg):
        if self.logEnabled == True:
            self.fileLog.write(msg)
            self.fileLog.write('\n')
            self.fileLog.flush()

    def countNumWordinDocument(self,matrix):
        ret = []
        for i in range(len(matrix)):
            wordInDoc = matrix[i]
            cnt = 0
            for j in range(len(wordInDoc)):
                cnt = cnt + wordInDoc[j]
            ret.append(cnt)
        return ret

    def logLDA(self,topN=10):
        if self.logEnabled == False:
            return

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
                self.logOut( (self.wordHeading[order[itr1]]+'('+str(value[itr1])+')') )
            self.logOut('')
        self.logOut( 'Document Prob')
        theta = []
        for d in range(self.intNumDoc):
            prob = zeros(shape=(self.intNumTopic),dtype=float)
            for n in range(self.numWordPerDoc[d]):
                idxWord = self.corpusList[d][n]
                for k in range(self.intNumTopic):
                    prob[k] = prob[k] + self.beta[k][idxWord]
            normalizeConstant = sum(prob)
            if normalizeConstant == 0:
                prob = ones(shape=(self.intNumTopic),dtype=float)
                normalizeConstant = sum(prob)
            for k in range(self.intNumTopic):
                prob[k] = prob[k] / normalizeConstant
            self.logOut( self.docHeading[d]+' : '+str(prob) )
            theta.append(prob)

        self.logOut('')
        self.logOut( 'Topic Prob')
        topicProportion = zeros(shape=(self.intNumTopic),dtype=float)
        for k in range(self.intNumTopic):
            for d in range(self.intNumDoc):
                topicProportion[k] = topicProportion[k] + theta[d][k]
        normalizeConstant = sum(topicProportion)
        if normalizeConstant == 0:
            topicProportion = ones(shape=(self.intNumTopic), dtype=float)
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
            print 'Topic ', k
            if topN > len(self.wordHeading):
                temptopN = len(self.wordHeading)
            else:
                temptopN = topN
            for itr1 in range(temptopN):
                print self.wordHeading[order[itr1]], '(', value[itr1], '),',
            print
        print 'Document Prob'
        theta = []
        for d in range(self.intNumDoc):
            prob = zeros(shape=(self.intNumTopic),dtype=float)
            for n in range(self.numWordPerDoc[d]):
                idxWord = self.corpusList[d][n]
                for k in range(self.intNumTopic):
                    prob[k] = prob[k] + self.beta[k][idxWord]
            normalizeConstant = sum(prob)
            if normalizeConstant == 0:
                prob = ones(shape=(self.intNumTopic),dtype=float)
                normalizeConstant = sum(prob)
            for k in range(self.intNumTopic):
                prob[k] = prob[k] / normalizeConstant
            print self.docHeading[d],' : ',prob
            theta.append(prob)

        print 'Topic Prob'
        topicProportion = zeros(shape=(self.intNumTopic),dtype=float)
        for k in range(self.intNumTopic):
            for d in range(self.intNumDoc):
                topicProportion[k] = topicProportion[k] + theta[d][k]
        normalizeConstant = sum(topicProportion)
        if normalizeConstant == 0:
            topicProportion = ones(shape=(self.intNumTopic), dtype=float)
            normalizeConstant = sum(topicProportion)
        for k in range(self.intNumTopic):
            topicProportion[k] = topicProportion[k] / normalizeConstant
        print topicProportion

    def calculateLogLikelihoodPerDocumentBlei(self,idxDoc):
        logLikelihood = 0.0

        d = idxDoc
        for k in range(self.intNumTopic):
            logLikelihood += (self.alpha[k]-1.0)*(-digamma(sum(self.gamma[d]))+digamma(self.gamma[d][k]))
        logLikelihood += gammaln(sum(self.alpha))
        for k in range(self.intNumTopic):
            logLikelihood += -gammaln(self.alpha[k])

        for n in range(self.numWordPerDoc[d]):
            for k in range(self.intNumTopic):
                logLikelihood += self.phi[d][n][k]*(-digamma(sum(self.gamma[d]))+digamma(self.gamma[d][k]))

        for n in range(self.numWordPerDoc[d]):
            for k in range(self.intNumTopic):
                logLikelihood += self.phi[d][n][k]*log(self.beta[k][self.corpusList[d][n]],e)

        for k in range(self.intNumTopic):
            logLikelihood += -(self.gamma[d][k] - 1.0)*(-digamma(sum(self.gamma[d])) + digamma(self.gamma[d][k]))
        logLikelihood -= gammaln(sum(self.gamma[d]))
        for k in range(self.intNumTopic):
            logLikelihood += -gammaln(self.gamma[d][k])

        for n in range(self.numWordPerDoc[d]):
            for k in range(self.intNumTopic):
                logLikelihood += self.phi[d][n][k]*log(self.phi[d][n][k],e)

        return logLikelihood

    def calculateLogLikelihoodBlei(self):
        logLikelihood = 0.0
        for d in range(self.intNumDoc):
            logLikelihood += self.calculateLogLikelihoodPerDocumentBlei(d)
        return logLikelihood

    def calculateLogLiklihoodMallet(self,testMatrixDataset,samplingReplication=3):
        avgLogLikelihood = 0.0
        stdLogLikelihood = 0.0
        corpusList = self.createBagOfWordList(testMatrixDataset.getMatrix())

        for itr in range(samplingReplication):
            logLikelihood = 0.0
            topicWordAssignmentInCorpus = zeros(shape=(self.intUniqueWord,self.intNumTopic),dtype=float)
            for d in range(len(testMatrixDataset.getRowHeading())):
                topicAssignmentOccurrenceInDoc = zeros(shape=(self.intNumTopic),dtype=float)
                for n in range(len(corpusList[d])):
                    currentWord = testMatrixDataset.getColHeading()[corpusList[d][n]]
                    if self.idxWordHash.has_key(currentWord) == True:
                        idxCurrentWord = self.idxWordHash[currentWord]
                        idxTopic = self.sampleTopicIndex(self.idxWordHash[currentWord])
                        topicAssignmentOccurrenceInDoc[idxTopic] = topicAssignmentOccurrenceInDoc[idxTopic] + 1
                        topicWordAssignmentInCorpus[idxCurrentWord][idxTopic] = topicWordAssignmentInCorpus[idxCurrentWord][idxTopic] + 1

                logLikelihood = logLikelihood + gammaln(sum(self.alpha))
                for k in range(self.intNumTopic):
                    logLikelihood = logLikelihood + gammaln(self.alpha[k]+topicAssignmentOccurrenceInDoc[k])
                    logLikelihood = logLikelihood - gammaln(self.alpha[k])
                logLikelihood = logLikelihood - gammaln(sum(self.alpha)+sum(topicAssignmentOccurrenceInDoc))

            nonZeroTopicAssignmentWord = 0
            for v in range(self.intUniqueWord):
                topicCounts = topicWordAssignmentInCorpus[v]
                for k in range(self.intNumTopic):
                    if topicCounts[k] == 0:
                        continue
                    nonZeroTopicAssignmentWord = nonZeroTopicAssignmentWord + 1
                    logLikelihood = logLikelihood + gammaln(self.beta[k][v]+topicCounts[k])

            for k in range(self.intNumTopic):
                sumBeta = 0
                sumToken = 0
                for v in range(self.intUniqueWord):
                    sumBeta = sumBeta + self.beta[k][v]
                    sumToken = sumToken + topicWordAssignmentInCorpus[v][k]
                logLikelihood = logLikelihood - gammaln(sumBeta + sumToken)

            avgLogLikelihood = avgLogLikelihood + logLikelihood
            stdLogLikelihood = stdLogLikelihood + logLikelihood*logLikelihood

        avgLogLikelihood = avgLogLikelihood / float(samplingReplication)
        stdLogLikelihood = sqrt(stdLogLikelihood/float(samplingReplication) - avgLogLikelihood*avgLogLikelihood)

        return avgLogLikelihood, stdLogLikelihood

    def sampleTopicIndex(self,idxWord):
        p = []
        for k in range(self.intNumTopic):
            p.append(self.beta[k][idxWord])
        sampledTopic = np.random.multinomial(1,p)
        for k in range(len(sampledTopic)):
            if sampledTopic[k] == 1:
                return k
                break
        return -1

    def performLDA(self):
        startTime = time.time()
        for iteration in range(self.numIterations):
            # Logging
            print('Variational Iteration : ' + str(iteration))
            print("Elapsed Time : " + str((time.time()-startTime)))# - startTime.microsecond) / 1e6))
            if self.logEnabled == True and iteration % self.logCycle == 1:
                logLikelihood = self.calculateLogLikelihoodBlei()
                print("Log Likelihood : " + str(logLikelihood))
                self.logOut('Variational Iteration : '+str(iteration))
                self.logOut("Elapsed Time : " +str((time.time()-startTime)))
                self.logOut("Log Likelihood : "+str(logLikelihood))

            # E-Step : Learning phi and gamma
            # initialize the variational parameter
            self.phi = []
            self.gamma = zeros(shape=(self.intNumDoc, self.intNumTopic), dtype=float)
            for d in range(self.intNumDoc):
                self.phi.append(zeros(shape=(self.numWordPerDoc[d], self.intNumTopic), dtype=float))
                for n in range(self.numWordPerDoc[d]):
                    for k in range(self.intNumTopic):
                        self.phi[d][n][k] = 1.0 / float(self.intNumTopic)
            for k in range(self.intNumTopic):
                for d in range(self.intNumDoc):
                    self.gamma[d][k] = self.alpha[k] + float(self.intUniqueWord) / float(self.intNumTopic)

            selectedDocsInBatch = np.random.randint(0,self.intNumDoc,self.sizeMiniBatch)
            for batchD in range(self.sizeMiniBatch):
                d = selectedDocsInBatch[batchD]
                for iterationInternal in range(self.numInternalIterations):
                    expDigammaGammaDK = zeros(shape=(self.intNumTopic),dtype=float)
                    for k in range(self.intNumTopic):
                        expDigammaGammaDK[k] = exp(digamma(self.gamma[d][k]))
                    # Learning phi
                    for n in range(self.numWordPerDoc[d]):
                        for k in range(self.intNumTopic):
                            self.phi[d][n][k] = self.beta[k][self.corpusList[d][n]] * expDigammaGammaDK[k]
                        normalizeConstantPhi = sum(self.phi[d][n])
                        for k in range(self.intNumTopic):
                            self.phi[d][n][k] = self.phi[d][n][k] / normalizeConstantPhi
                    # Learning gamma
                    for k in range(self.intNumTopic):
                        self.gamma[d][k] = self.alpha[k]
                        for n in range(self.numWordPerDoc[d]):
                            self.gamma[d][k] = self.gamma[d][k] + self.phi[d][n][k]

            # M-Step : Learning alpha and beta
            # Learning Beta
            tempBeta = zeros(shape=(self.intNumTopic,self.intUniqueWord),dtype=float)
            for k in range(self.intNumTopic):
                for v in range(self.intUniqueWord):
                    tempBeta[k][v] = self.beta[k][v]
            for k in range(self.intNumTopic):
                for batchD in range(self.sizeMiniBatch):
                    d = selectedDocsInBatch[batchD]
                    for n in range(self.numWordPerDoc[d]):
                        tempBeta[k][self.corpusList[d][n]] = tempBeta[k][self.corpusList[d][n]] + float(self.intNumDoc/self.sizeMiniBatch) * self.phi[d][n][k]
                normalizeConstantBeta = sum(tempBeta[k])
                for v in range(self.intUniqueWord):
                    tempBeta[k][v] = tempBeta[k][v] / normalizeConstantBeta
            for k in range(self.intNumTopic):
                for v in range(self.intUniqueWord):
                    self.beta[k][v] = (1.0 - self.learningRate) * self.beta[k][v] + self.learningRate * tempBeta[k][v]

            # Learning Alpha
            # calculate current ELBO with respect to Current Alpha
            ELBOMax = 0
            for batchD in range(self.sizeMiniBatch):
                d = selectedDocsInBatch[batchD]
                ELBOMax = ELBOMax + gammaln(sum(self.alpha))
                for k in range(self.intNumTopic):
                    ELBOMax = ELBOMax - gammaln(self.alpha[k])
                    ELBOMax = ELBOMax + float(self.intNumDoc/self.sizeMiniBatch) * (self.alpha[k] - 1) * (
                        digamma(self.gamma[d][k]) - digamma(sum(self.gamma[d])))
            tempAlpha = zeros(shape=(self.intNumTopic), dtype=float)
            bestAlpha = zeros(shape=(self.intNumTopic), dtype=float)
            for k in range(self.intNumTopic):
                bestAlpha[k] = self.alpha[k]
                tempAlpha[k] = self.alpha[k]

            # self.logOut( 'Newton-Rhapson Itr - ELBO : '+str(ELBOMax)+' | alpha : '+str(tempAlpha) )
            # Newton-Rhapson optimization
            for itr in range(self.numNewtonIteration):
                # Building Hessian Matrix and Derivative Vector
                H = zeros(shape=(self.intNumTopic, self.intNumTopic), dtype=float)
                g = zeros(shape=(self.intNumTopic), dtype=float)
                for k1 in range(self.intNumTopic):
                    g[k1] = float(self.intNumDoc) * (digamma(sum(tempAlpha)) - digamma(tempAlpha[k1]))
                    for batchD in range(self.sizeMiniBatch):
                        d = selectedDocsInBatch[batchD]
                        g[k1] = g[k1] + float(self.intNumDoc/self.sizeMiniBatch) * (digamma(self.gamma[d][k1]) - digamma(sum(self.gamma[d])))
                    for k2 in range(self.intNumTopic):
                        H[k1][k2] = 0
                        if k1 == k2:
                            H[k1][k2] = H[k1][k2] - float(self.sizeMiniBatch) * polygamma(1, tempAlpha[k1])
                        H[k1][k2] = H[k1][k2] + float(self.sizeMiniBatch) * polygamma(1, sum(tempAlpha))

                # Update Alpha in Log Domain
                deltaAlpha = np.dot(np.linalg.inv(H), g)

                for k in range(self.intNumTopic):
                    if deltaAlpha[k] > 10.0:
                        deltaAlpha[k] = 10.0
                    if deltaAlpha[k] < -10.0:
                        deltaAlpha[k] = -10.0
                    logAlphaK = log(tempAlpha[k], e)
                    logAlphaK = logAlphaK - deltaAlpha[k]
                    tempAlpha[k] = exp(logAlphaK)
                    if tempAlpha[k] > 100.0:
                        tempAlpha[k] = 100.0
                    if tempAlpha[k] < 0.00001:
                        tempAlpha[k] = 0.00001

                # calculate current ELBO with respect to New Alpha
                ELBOAfter = 0
                for batchD in range(self.sizeMiniBatch):
                    d = selectedDocsInBatch[batchD]
                    ELBOAfter = ELBOAfter + gammaln(sum(tempAlpha))
                    for k in range(self.intNumTopic):
                        ELBOAfter = ELBOAfter - gammaln(tempAlpha[k])
                        ELBOAfter = ELBOAfter + float(self.intNumDoc/self.sizeMiniBatch) * (tempAlpha[k] - 1) * (digamma(self.gamma[d][k]) - digamma(sum(self.gamma[d])))

                # self.logOut( 'Newton-Rhapson Itr - ELBO : '+str(ELBOAfter)+' | alpha : '+str(tempAlpha) )

                if ELBOMax <= ELBOAfter:
                    ELBOMax = ELBOAfter
                    for k in range(self.intNumTopic):
                        bestAlpha[k] = tempAlpha[k]

            for k in range(self.intNumTopic):
                self.alpha[k] = (1.0 - self.learningRate) * self.alpha[k] + self.learningRate * bestAlpha[k]

            self.learningRate = self.learningRate * self.discountInLearningRatePerItr



if __name__ == "__main__":
    #reader = BagOfWordReader('../../TestDataset/Sample-bow.csv','../../TestDataset/Sample-words.csv')
    #reader = BagOfWordReader('../../TestDataset/SentimentDataset-bow.csv','../../TestDataset/SentimentDataset-words.csv')
    reader = SparseBagOfWordReader('../../TestDataset/ap.dat', '../../TestDataset/vocab.txt')

    matrixData = reader.getResult()

    filter = BagOfWordFilter(matrixData)
    matrixData = filter.filterDF(0.30,0.70)
    filter2 = BagOfWordFilter(matrixData)
    matrixData = filter2.filterShortWord(3)

    print 'Num of word : ',len(matrixData.getColHeading())
    print 'Num of doc : ' ,len(matrixData.getRowHeading())
    LDA = LDA_StochasticVariational(matrixData,10,2000,numLocalIterations=30,sizeMiniBatch=200,logCycle=10,logEnabled=True,strLogFilename='./log/log_svi.txt')
    print 'Finished Initialization....'
    LDA.performLDA()
    print 'Finished LDA-VI....'
    LDA.printLDA(5)

