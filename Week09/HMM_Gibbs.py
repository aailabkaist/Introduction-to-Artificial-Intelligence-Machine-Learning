import random
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import gridspec

class HMMGibbs:
    def plotPoints(self,data,affilitation,means,covs,affilitationProb):
        plt.figure(1)

        gs = gridspec.GridSpec(3,1)
        axarr0 = plt.subplot(gs[:2,:])

        types = []
        for i in range(len(affilitation)):
            if affilitation[i] in types:
                pass
            else:
                types.append(affilitation[i])

        for i in range(len(data)):
            if i >= 1:
                axarr0.arrow(data[i-1][0],data[i-1][1],data[i][0]-data[i-1][0],data[i][1]-data[i-1][1],head_width=0.1,head_length=0.15,fc='k',ec='k',length_includes_head=True)

        colors = [ 'r', 'g', 'b', 'y', 'k']
        totalX = []
        totalY = []
        for j in range(len(types)):
            x = []
            y = []
            for i in range(len(data)):
                if affilitation[i] == types[j]:
                    x.append(data[i][0])
                    y.append(data[i][1])
                    totalX.append(data[i][0])
                    totalY.append(data[i][1])
            axarr0.plot(x,y,colors[j%len(colors)]+'o')

        gridX = np.arange(min(totalX)-0.1*(max(totalX)-min(totalX)),max(totalX)+0.1*(max(totalX)-min(totalX)), (max(totalX)-min(totalX))/100)
        gridY = np.arange(min(totalY)-0.1*(max(totalY)-min(totalY)),max(totalY)+0.1*(max(totalY)-min(totalY)), (max(totalY)-min(totalY))/100)
        meshX, meshY = np.meshgrid(gridX,gridY)

        for j in range(len(types)):
            Z = np.zeros(shape=(len(gridY),len(gridX)),dtype=float)
            for itr1 in range(len(meshX)):
                for itr2 in range(len(meshX[itr1])):
                    Z[itr1][itr2] = stats.multivariate_normal.pdf( [meshX[itr1][itr2],meshY[itr1][itr2]], mean=means[j], cov=covs[j])
            CS = axarr0.contour(meshX,meshY,Z,3,colors='k')
            axarr0.clabel(CS,inline=1,fontsize=10)

        axarr1 = plt.subplot(gs[2, :])

        affilitationProbTranspose = []
        for i in range(len(types)):
            temp = []
            for j in range(len(data)):
                temp.append(affilitationProb[j][i])
            affilitationProbTranspose.append(temp)

        temp = np.zeros(len(data))
        for i in range(len(types)):
            axarr1.bar(range(len(data)),affilitationProbTranspose[i],width=1.0,bottom=temp,color=colors[i],edgecolor="none")
            for j in range(len(data)):
                temp[j] = temp[j] + affilitationProbTranspose[i][j]

        plt.tight_layout()
        #plt.savefig('./log/hmm_plot.png')

    def initialize(self,data,k):
        transition = []
        initial = []
        for i in range(k):
            temp = []
            initial.append(1.0/float(k))
            for j in range(k):
                temp.append(1.0/float(k))
            transition.append(temp)

        dataCluster = []
        for j in range(k):
            dataCluster.append([])

        affilitationProb = []
        estimatedLabel = []
        for i in range(len(data)):
            temp = []
            for j in range(k):
                temp.append(0.0)
            affilitationProb.append(temp)

            idx = random.randrange(0,k)
            dataCluster[idx].append(data[i])
            affilitationProb[i][idx] = 1.0
            estimatedLabel.append(idx)

        means = []
        for j in range(k):
            means.append(np.mean(dataCluster[j],axis=0))

        covs = []
        for j in range(k):
            covs.append(np.cov(np.array(dataCluster[j]).T))

        return means,covs,transition,initial,affilitationProb,estimatedLabel

    def inferenceSampling(self,k,data,itr):
        means,covs,transition,initial,affilitationProb,estimatedLabel = self.initialize(data,k)

        for i in range(itr):
            #print ('------------------------------------------------------------------------------------')
            #print ('Iteration : ', i)
            #print ('------------------------------------------------------------------------------------')
            #print ('Means : ' + str(means))
            #print ('Covs : ' + str(covs))
            #print ('Affilitation Prob. : ' + str(affilitationProb))

            for j in range(len(data)):
                estimatedLabel[j],affilitationProb[j] = self.sampleLabel(k,data[j],means,covs,transition,initial,estimatedLabel,j,len(data))
                means, covs, transition, initial = self.learningParameters(data,estimatedLabel,affilitationProb,k)

            #self.plotPoints(data, estimatedLabel, means, covs, affilitationProb)
        return means, covs, transition, affilitationProb, estimatedLabel

    def learningParameters(self,data,estimatedLabel,affilitationProb,k):
        initial = []
        for i in range(k):
            initial.append(0.001)
        initial[int(estimatedLabel[0])] = 1.0
        normalize = 0.0
        for i in range(k):
            normalize = normalize + initial[i]
        for i in range(k):
            initial[i] = initial[i] / normalize

        transition = []
        for i in range(k):
            temp = []
            for j in range(k):
                temp.append(0.001)
            transition.append(temp)
        for i in range(len(data)-1):
            transition[int(estimatedLabel[i])][int(estimatedLabel[i+1])] = transition[int(estimatedLabel[i])][int(estimatedLabel[i+1])] + 1.0

        for i in range(k):
            normalize = 0.0
            for j in range(k):
                normalize = normalize + transition[i][j]
            for j in range(k):
                transition[i][j] = transition[i][j] / normalize

        means = []
        firstindex = 0
        for j in range(k):
            temp = data[firstindex] * affilitationProb[firstindex][j]
            normalize = 0.0
            for i in range(1,len(data)):
                temp = temp + data[i] * affilitationProb[i][j]
                normalize = normalize + affilitationProb[i][j]
            temp = temp / normalize
            means.append(temp)

        covs = []
        for j in range(k):
            temp = (data[0] - means[j]) * (data[0] - means[j]) * affilitationProb[0][j]
            for i in range(1,len(data)):
                temp = temp + np.outer((data[i] - means[j]),(data[i] - means[j])) * affilitationProb[i][j]
                normalize = normalize + affilitationProb[i][j]
            temp = temp / normalize
            covs.append(temp)

        return means,covs,transition,initial

    def sampleLabel(self,k,instance,means,covs,transition,initial,estimatedLabel,j,length):
        loglikelihood = []
        likelihood = []
        normalize = 0
        for i in range(k):
            if length == j+1:
                logLikelihoodNextState = 0
            else:
                logLikelihoodNextState = np.log(transition[i][int(estimatedLabel[j+1])])
            if j == 0:
                logLikelihoodPrevState = np.log(initial[i])
            else:
                logLikelihoodPrevState = np.log(transition[int(estimatedLabel[j-1])][i])
            covs[i] = covs[i] + 1.0 * np.array([[1.0, 0.0], [0.0, 1.0]])

            logLikelihoodObservation = np.log(stats.multivariate_normal(means[i],covs[i]).pdf(instance))
            loglikelihood.append(logLikelihoodNextState+logLikelihoodPrevState+logLikelihoodObservation)
            likelihood.append(np.exp(loglikelihood[i]) + 0.001)
            normalize = normalize + likelihood[i]


        for i in range(k):
            likelihood[i] = likelihood[i] / normalize

        sample = np.random.multinomial(1,likelihood,size=1)
        ret = -1
        for i in range(k):
            if sample[0][i] > 0.5:
                ret = i
        return ret,likelihood

if __name__=="__main__":
    random.seed(0)
    np.random.seed(0)

    trueChange = np.concatenate( ( np.zeros(20) , np.ones(30), np.ones(40)*2.0 , np.zeros(10) , np.ones(20) ) , axis = 0)
    trueAffilitationProb = []
    for i in range(len(trueChange)):
        temp = []
        for j in range(3):
            if trueChange[i] == j:
                temp.append(1)
            else:
                temp.append(0)
        trueAffilitationProb.append(temp)

    trueMean = [ [2,4], [-1,3], [0,0] ]
    #trueCov = [ [[0.9,0.5],[0.5,0.9]] , [[0.4,0.6],[0.5,0.9]], [[1.0,-0.5],[-0.5,1.0]] ]
    trueCov = [[[0.45, 0.25], [0.25, 0.45]], [[0.2, 0.3], [0.25, 0.45]], [[0.5, -0.25], [-0.25, 0.5]]]

    data = []
    for i in range(len(trueChange)):
        data.append(np.random.multivariate_normal(trueMean[int(trueChange[i])],trueCov[int(trueChange[i])],1)[0])

    print ( "True Chage : ",trueChange)
    print ( "Data : ",data)

    hmm = HMMGibbs()
    hmm.plotPoints(data,trueChange,trueMean,trueCov,trueAffilitationProb)
    hmm.inferenceSampling(3,data,10)