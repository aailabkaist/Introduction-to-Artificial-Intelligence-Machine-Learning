import tensorflow as tf
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

class LDA_TF_Variational:

    intNumTopic = 0
    numIterations = 0
    numInternalIterations = 10
    numNewtonIteration = 10

    intNumDoc = 0
    intUniqueWord = 0

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

    def __init__(self,matrixDataset,intNumTopic,numIterations,strLogFilename='log.txt',logEnabled=False,logCycle=10,numLocalIterations=10):
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
                prob = ones(shape=(self.intNumTopic), dtype=float)
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
            print 'Topic ',k
            if topN > len(self.wordHeading):
                temptopN = len(self.wordHeading)
            else:
                temptopN = topN
            for itr1 in range(temptopN):
                print self.wordHeading[order[itr1]],'(',value[itr1],'),',
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

        tfAlphas = []
        for k in range(self.intNumTopic):
            tfAlphas.append( tf.Variable(self.alpha[k],dtype=tf.float64) )
        tfBetas = []
        for k in range(self.intNumTopic):
            tfBetas.append([])
            for v in range(self.intUniqueWord):
                tfBetas[k].append( tf.Variable(self.beta[k][v],dtype=tf.float64) )

        tfPhi = []
        tfGamma = []
        self.phi = []
        self.gamma = []
        for d in range(self.intNumDoc):
            tfPhi.append([])
            self.phi.append([])
            for n in range(self.numWordPerDoc[d]):
                tfPhi[d].append([])
                self.phi[d].append([])
                for k in range(self.intNumTopic):
                    tfPhi[d][n].append( tf.Variable(1.0 / float(self.intNumTopic),dtype=tf.float64 ) )
                    self.phi[d][n].append(0)

        for d in range(self.intNumDoc):
            tfGamma.append([])
            self.gamma.append([])
            for k in range(self.intNumTopic):
                tfGamma[d].append( tf.Variable(self.alpha[k] + float(self.intUniqueWord) / float(self.intNumTopic),dtype=tf.float64) )
                self.gamma[d].append(0)

        ELBO = tf.constant(0.0,dtype=tf.float64)
        sumTFAlpha = tf.constant(0.0,dtype=tf.float64)
        for k in range(self.intNumTopic):
            sumTFAlpha = tf.add(sumTFAlpha,tfAlphas[k])

        V = self.intUniqueWord
        maxDocLength = -1
        for d in range(self.intNumDoc):
            if maxDocLength < self.numWordPerDoc[d]:
                maxDocLength = self.numWordPerDoc[d]
        tfBOW = tf.placeholder(tf.float64, [self.intNumDoc,maxDocLength,V])

        for d in range(self.intNumDoc):
            N = self.numWordPerDoc[d]

            sumTFGamma = tf.constant(0.0,dtype=tf.float64)
            for k in range(self.intNumTopic):
                sumTFGamma = tf.add(sumTFGamma,tfGamma[d][k])

            # Expectation of P(theta|alpha) given q
            for k in range(self.intNumTopic):
                term1 = tf.sub(tfAlphas[k],tf.constant(1,dtype=tf.float64))
                term2 = tf.sub(tf.digamma(tfGamma[d][k]),tf.digamma(sumTFGamma))
                term3 = tf.mul(term1,term2)
                ELBO = tf.add(ELBO,term3)
            ELBO = tf.add(ELBO,tf.lgamma(sumTFAlpha))
            for k in range(self.intNumTopic):
                ELBO = tf.sub(ELBO,tf.lgamma(tfAlphas[k]))

            # Expectation of P(z|theta) given q
            for n in range(N):
                for k in range(self.intNumTopic):
                    term4 = tf.sub(tf.digamma(tfGamma[d][k]),tf.digamma(sumTFGamma))
                    term5 = tf.mul(tfPhi[d][n][k],term4)
                    ELBO = tf.add(ELBO,term5)


            # Expectation of P(w|z,beta) given q
            for n in range(N):
                for k in range(self.intNumTopic):
                    for v in range(self.intUniqueWord):
                        term6 = tf.mul(tf.slice(tfBOW, [d, n, v], [1, 1, 1]),tf.log(tfBetas[k][v]))
                        term7 = tf.mul(tfPhi[d][n][k],term6)
                        ELBO = tf.add(ELBO,term7)

            # Entropy of q
            for k in range(self.intNumTopic):
                term8 = tf.sub(tf.constant(1,dtype=tf.float64),tfGamma[d][k])
                term9 = tf.sub(tf.digamma(tfGamma[d][k]),tf.digamma(sumTFGamma))
                term10 = tf.mul(term8,term9)
                ELBO = tf.add(ELBO,term10)
            ELBO = tf.sub(ELBO,tf.lgamma(sumTFGamma))
            for k in range(self.intNumTopic):
                ELBO = tf.add(ELBO,tf.lgamma(tfGamma[d][k]))

            for n in range(N):
                for k in range(self.intNumTopic):
                    term11 = tf.mul(tfPhi[d][n][k],tf.log(tfPhi[d][n][k]))
                    ELBO = tf.sub(ELBO,term11)

            # Adding constraints of Phi as Prob Simplex
            for n in range(N):
                sumTFPhiDN = tf.constant(0.0, dtype=tf.float64)
                for k in range(self.intNumTopic):
                    sumTFPhiDN = tf.add(sumTFPhiDN, tfPhi[d][n][k])
                sumTFPhiDN = tf.sub( sumTFPhiDN , tf.constant(1.0,dtype=tf.float64) )
                lagrangeMultiplierPhi = tf.Variable(1.0,dtype=tf.float64)
                lagrangeTermPhi = tf.mul(lagrangeMultiplierPhi,sumTFPhiDN)
                ELBO = tf.add(ELBO,lagrangeTermPhi)

        # Adding constraints of Phi as Prob Simplex
        for k in range(self.intNumTopic):
            sumTFBetaK = tf.constant(0.0, dtype=tf.float64)
            for v in range(V):
                sumTFBetaK = tf.add(sumTFBetaK, tfBetas[k][v])
            sumTFBetaK = tf.sub(sumTFBetaK, tf.constant(1.0, dtype=tf.float64))
            lagrangeMultiplierBeta = tf.Variable(1.0, dtype=tf.float64)
            lagrangeTermBeta = tf.mul(lagrangeMultiplierBeta, sumTFPhiDN)
            ELBO = tf.add(ELBO, lagrangeTermBeta)

        ELBO = tf.mul(tf.constant(-1,dtype=tf.float64),ELBO)
        training = tf.train.GradientDescentOptimizer(0.01).minimize(ELBO)

        # Session initialization
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        init = tf.initialize_all_variables()
        sess.run(init)

        BOW = []
        for d in range(self.intNumDoc):
            BOW.append([])
            for n in range(self.numWordPerDoc[d]):
                BOW[d].append([])
                for v in range(self.intUniqueWord):
                    if self.corpusList[d][n] == v:
                        BOW[d][n].append(1)
                    else:
                        BOW[d][n].append(0)
            for itr in range(self.numWordPerDoc[d],maxDocLength):
                BOW[d].append([])
                for v in range(self.intUniqueWord):
                    BOW[d][itr].append(0)

        # Session Running for Training
        for i in range(30):
            sess.run(training, feed_dict={tfBOW:BOW})
            for k in range(self.intNumTopic):
                self.alpha[k] = sess.run(tfAlphas[k], feed_dict={tfBOW:BOW})
            for k in range(self.intNumTopic):
                normalize = 0.0
                for v in range(self.intUniqueWord):
                    self.beta[k][v] = sess.run(tfBetas[k][v], feed_dict={tfBOW:BOW})
                    normalize = normalize + self.beta[k][v]
                for v in range(self.intUniqueWord):
                    self.beta[k][v] = self.beta[k][v] / normalize
            for d in range(self.intNumDoc):
                for n in range(self.numWordPerDoc[d]):
                    normalize = 0.0
                    for k in range(self.intNumTopic):
                        self.phi[d][n][k] = sess.run(tfPhi[d][n][k], feed_dict={tfBOW:BOW})
                        normalize = normalize + self.phi[d][n][k]
                    for k in range(self.intNumTopic):
                        self.phi[d][n][k] = self.phi[d][n][k] / normalize
            for d in range(self.intNumDoc):
                for k in range(self.intNumTopic):
                    self.gamma[d][k] = sess.run(tfGamma[d][k], feed_dict={tfBOW:BOW})
            print "Iteration ",i
            print "ELBO : ",sess.run(ELBO, feed_dict={tfBOW:BOW})

        sess.close()

if __name__ == "__main__":
    #reader = BagOfWordReader('../../TestDataset/Sample-bow.csv','../../TestDataset/Sample-words.csv')
    #reader = BagOfWordReader('../../TestDataset/SentimentDataset-bow.csv','../../TestDataset/SentimentDataset-words.csv')
    reader = SparseBagOfWordReader('../../TestDataset/ap.dat', '../../TestDataset/vocab.txt')

    matrixData = reader.getResult()

    filter = BagOfWordFilter(matrixData)
    matrixData = filter.filterDF(0.0,1.0)
    filter2 = BagOfWordFilter(matrixData)
    matrixData = filter2.filterShortWord(0)

    print 'Num of word : ',len(matrixData.getColHeading())
    print 'Num of doc : ' ,len(matrixData.getRowHeading())
    LDA = LDA_TF_Variational(matrixData,2,100,logEnabled=True,strLogFilename='./log/log_tf_vi.txt')
    print 'Finished Initialization....'
    LDA.performLDA()
    print 'Finished LDA-VI....'
    LDA.printLDA(5)

