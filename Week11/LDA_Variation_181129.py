import time
import numpy as np
from numpy import ndarray, sum
from scipy.special import polygamma, gamma as gammaFunc
from math import log, e, exp
import csv
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

def loggammaFunc(num):
    result = 0
    if num < 170:
        result = log(gammaFunc(num), e)
    else:
        result = log(num-1, e) + loggammaFunc(num-1)
    return result

class LDA_Variational:

    intNumTopic = 0
    numIterations = 0
    numInternalIterations = 10
    numNewtonIteration = 5

    intNumDoc = 0
    intUniqueWord = 0
    word = []
    stopword = []
    numWordPerDoc = []
    corpusMatrix = []
    corpusList = []

    alpha = []
    beta = []

    fileLog = 0

    def __init__(self,strBagOfWordFileName,strLogFileName,intNumTopic,numIterations):
        self.readBagOfWord(strBagOfWordFileName)
        self.makeCorpusList()                       # make a corpuslist using "updated" word
        self.countNumWordinDocument()
        self.numIterations = numIterations
        #self.numInternalIterations = numIterations
        self.intNumDoc = len(self.corpusMatrix)
        self.intUniqueWord = len(self.corpusMatrix[0])
        self.intNumTopic = intNumTopic
        self.fileLog = open(strLogFileName,'w')
        print("Complete initial settings!")

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
        #self.logLDA()
        self.fileLog.close()

    def logOut(self,msg):
        self.fileLog.write(msg)
        self.fileLog.write('\n')
        self.fileLog.flush()

    def readBagOfWord(self,strFileName):
        cats = ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
                'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
        newsgroups_train = fetch_20newsgroups(subset='train',shuffle=True, random_state=0, categories=cats) # load "train" dataset

        bagVocab = np.load(strFileName)
        target = newsgroups_train.target  # topic of each news

        vect = CountVectorizer(vocabulary=bagVocab, binary=True)  # the way of count words (in bagVocab)
        data = vect.fit_transform(newsgroups_train.data)  # count words in each news (in bagVocab)
        self.corpusMatrix = data.toarray().tolist()
        self.word = bagVocab.tolist()

        #newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=0)
        #data_test = vect.fit_transform(newsgroups_test.data)
        #target_test = newsgroups_test.target

    def makeCorpusList(self):
        for i in range(len(self.corpusMatrix)):
            listInCorpus = []
            for j in range(len(self.corpusMatrix[0])):
                if self.corpusMatrix[i][j] == 1:    # if jth unique word in ith document, append index of w in ith list in corpusList
                    listInCorpus.append(j)
            self.corpusList.append(listInCorpus)


    def countNumWordinDocument(self):
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
        print ('Alpha')
        print (self.alpha)
        print ('Beta')
        for k in range(self.intNumTopic):
            print (self.beta[k])
        print ('Topic Words')
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
            print ('Topic ',k)
            for itr1 in range(topN):
                print (self.word[order[itr1]],'(',value[itr1],'),',)

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
            theta.append(prob)

        print ('Topic Prob')
        topicProportion = ndarray(shape=(self.intNumTopic),dtype=float)
        for k in range(self.intNumTopic):
            for d in range(self.intNumDoc):
                topicProportion[k] = topicProportion[k] + theta[d][k]
        normalizeConstant = sum(topicProportion)
        for k in range(self.intNumTopic):
            topicProportion[k] = topicProportion[k] / normalizeConstant
        print (topicProportion)

    def performLDA(self):

        start_time = time.time()

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
                gamma[d][k] = self.alpha[k] + float(self.numWordPerDoc[d]) / float(self.intNumTopic)

        for iteration in range(self.numIterations):
            print ('Iteration : ' + str(iteration+1) + ' (start time = ' + str(round(time.time()-start_time,2)) + ')')
            #self.logLDA()

            self.logOut( 'Variational Iteration : '+str(iteration+1) )

            # E-Step : Learning phi and gamma
            for iterationInternal in range(self.numInternalIterations):
                # Learning phi
                for d in range(self.intNumDoc):
                    for n in range(self.numWordPerDoc[d]):
                        for k in range(self.intNumTopic):
                            phi[d][n][k] = self.beta[k][self.corpusList[d][n]]*exp(polygamma(0,gamma[d][k]))
                        normalizeConstantPhi = sum(phi[d][n])
                        for k in range(self.intNumTopic):
                            phi[d][n][k] = float(phi[d][n][k]) / normalizeConstantPhi
                # Learning gamma
                for d in range(self.intNumDoc):
                    for k in range(self.intNumTopic):
                        gamma[d][k] = self.alpha[k]
                        for n in range(self.numWordPerDoc[d]):
                            gamma[d][k] = gamma[d][k] + phi[d][n][k]

            # M-Step : Learning alpha and beta

            # Learning Beta
            for k in range(self.intNumTopic):
                for v in range(self.intUniqueWord):
                    self.beta[k][v] = 0
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
            # E_q(logP(theta|alpha))
            for d in range(self.intNumDoc):
                ELBOMax = ELBOMax + loggammaFunc(sum(self.alpha))
                for k in range(self.intNumTopic):
                    ELBOMax = ELBOMax - loggammaFunc(self.alpha[k])
                    ELBOMax = ELBOMax + (self.alpha[k] - 1) * (polygamma(0, gamma[d][k]) - polygamma(0, sum(gamma[d])))
            print(ELBOMax)
            # E_q(logP(z|theta))
            for d in range(self.intNumDoc):
                for n in range(self.numWordPerDoc[d]):
                    for k in range(self.intNumTopic):
                        ELBOMax = ELBOMax + phi[d][n][k] * (polygamma(0, gamma[d][k]) - polygamma(0, sum(gamma[d])))
            print(ELBOMax)
            # E_q(logP(w|z, beta))
            for d in range(self.intNumDoc):
                for n in range(self.numWordPerDoc[d]):
                    for k in range(self.intNumTopic):
                        ELBOMax = ELBOMax + phi[d][n][k] * log(self.beta[k][self.corpusList[d][n]], e)
            print(ELBOMax)

            # H(q)
            for d in range(self.intNumDoc):
                ELBOMax = ELBOMax - loggammaFunc(sum(gamma[d]))
                for k in range(self.intNumTopic):
                    ELBOMax = ELBOMax + loggammaFunc(gamma[d][k])
                    ELBOMax = ELBOMax - (gamma[d][k] - 1) * (polygamma(0, gamma[d][k]) - polygamma(0, sum(gamma[d])))
                    for n in range(self.numWordPerDoc[d]):
                        ELBOMax = ELBOMax - phi[d][n][k] * log(phi[d][n][k])
            print(ELBOMax)

            tempAlpha = ndarray(shape=(self.intNumTopic), dtype=float)
            bestAlpha = ndarray(shape=(self.intNumTopic), dtype=float)
            for k in range(self.intNumTopic):
                bestAlpha[k] = self.alpha[k]
                tempAlpha[k] = self.alpha[k]
            self.logOut( 'Newton-Rhapson Itr - ELBO : '+str(ELBOMax)+' | alpha : '+str(tempAlpha) )
            print('Newton-Rhapson Itr - ELBO : '+str(ELBOMax)+' | alpha : '+str(tempAlpha))

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

                # Update Alpha in Domain
                deltaAlpha = np.dot(np.linalg.inv(H),g)

                for k in range(self.intNumTopic):
                    tempAlpha[k] = tempAlpha[k] - deltaAlpha[k]
                    #logAlphaK = log(tempAlpha[k],e)
                    #logAlphaK = logAlphaK - deltaAlpha[k]
                    #tempAlpha[k] = exp(logAlphaK)
                    if tempAlpha[k] < 0.00001:
                        tempAlpha[k] = 0.00001

                # calculate current ELBO with respect to New Alpha
                ELBOAfter = 0
                # E_q(logP(theta|alpha))
                for d in range(self.intNumDoc):
                    ELBOAfter = ELBOAfter + loggammaFunc(sum(tempAlpha))
                    for k in range(self.intNumTopic):
                        ELBOAfter = ELBOAfter - loggammaFunc(tempAlpha[k])
                        ELBOAfter = ELBOAfter + (tempAlpha[k] - 1) * (polygamma(0, gamma[d][k]) - polygamma(0, sum(gamma[d])))
                print(ELBOAfter)
                # E_q(logP(z|theta))
                for d in range(self.intNumDoc):
                    for n in range(self.numWordPerDoc[d]):
                        for k in range(self.intNumTopic):
                            ELBOAfter = ELBOAfter + phi[d][n][k] * (polygamma(0, gamma[d][k]) - polygamma(0, sum(gamma[d])))
                print(ELBOAfter)
                # E_q(logP(w|z, beta))
                for d in range(self.intNumDoc):
                    for n in range(self.numWordPerDoc[d]):
                        for k in range(self.intNumTopic):
                            ELBOAfter = ELBOAfter + phi[d][n][k] * log(self.beta[k][self.corpusList[d][n]], e)
                print(ELBOAfter)

                # H(q)
                for d in range(self.intNumDoc):
                    ELBOAfter = ELBOAfter - loggammaFunc(sum(gamma[d]))
                    for k in range(self.intNumTopic):
                        ELBOAfter = ELBOAfter + loggammaFunc(gamma[d][k])
                        ELBOAfter = ELBOAfter - (gamma[d][k] - 1) * (polygamma(0, gamma[d][k]) - polygamma(0, sum(gamma[d])))
                        for n in range(self.numWordPerDoc[d]):
                            ELBOAfter = ELBOAfter - phi[d][n][k] * log(phi[d][n][k])
                print(ELBOAfter)

                self.logOut( 'Newton-Rhapson Itr - ELBO : '+str(ELBOAfter)+' | alpha : '+str(tempAlpha) )
                print('Newton-Rhapson Itr - ELBO : '+str(ELBOAfter)+' | alpha : '+str(tempAlpha) )

                #if ELBOMax <= ELBOAfter:
                #    ELBOMax = ELBOAfter
                #    for k in range(self.intNumTopic):
                #        bestAlpha[k] = tempAlpha[k]
                for k in range(self.intNumTopic):
                    bestAlpha[k] = tempAlpha[k]

            self.alpha = bestAlpha

            with open("ouput_gamma"+str(iteration+1)+".csv", "wb") as f:
                writer = csv.writer(f)
                writer.writerows(gamma)

print("start!")


#LDA = LDA_Variational('Sample-bow.csv','Sample-words.csv','log.txt',2,100)
LDA = LDA_Variational('bagVocab_3+_newstop_tfidf_2.npy','log.txt',5,3)
LDA.performLDA()
#LDA.printLDA(5)

arrMat = np.array(LDA.beta)
arrWord = np.array(LDA.word)

num_words = 20

print("Top ",'%2d'%(num_words), " Words in Each Topic\n")

for i in range(LDA.intNumTopic):
    index = (-arrMat[i]).argsort()
    top20idx = index[:num_words]
    s = 'Topic %2d : '%(i + 1)
    for word in arrWord[top20idx]:
        s += ' %s,' % (word)
    print(s[:-1])