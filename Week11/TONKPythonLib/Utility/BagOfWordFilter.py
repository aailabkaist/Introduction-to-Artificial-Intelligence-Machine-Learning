from numpy import ndarray, sum
from math import log, e
from BagOfWordReader import BagOfWordReader
from TONKPythonLib.Dataset.MatrixDataset import MatrixDataset

class BagOfWordFilter:

    matrixData = []
    rowHeading = []
    colHeading = []

    DF = [] # ndarray size : numWord
    TFIDF = [] # ndarray size : numDoc X numWord

    numWord = 0
    numDoc = 0

    # matrix : Row(Document) X Col(Word), cell : count of word appearance in document
    def __init__(self, matrixData):
        self.matrixData = matrixData
        self.colHeading = self.matrixData.getColHeading()
        self.rowHeading = self.matrixData.getRowHeading()

        self.numDoc = len(self.rowHeading)
        self.numWord = len(self.colHeading)

        self.DF = ndarray(shape=(self.numWord),dtype=float)
        self.TFIDF = ndarray(shape=(self.numDoc,self.numWord), dtype=float)

        TF = ndarray(shape=(self.numDoc, self.numWord), dtype=float)
        IDF = ndarray(shape=(self.numWord), dtype=float)

        for i in range(self.numDoc):
            idxNonZeroWord = self.matrixData.getNonZeroVectorElementIdx(i)
            for j in range(len(idxNonZeroWord)):
                TF[i][j] = 1 + log(self.matrixData.getMatrixValue(i,idxNonZeroWord[j]),e)
                self.DF[j] = self.DF[j] + 1

        for i in range(self.numWord):
            IDF[i] = log( float(self.numDoc) / (self.DF[i] + 1.0), e)

        for i in range(self.numDoc):
            for j in range(self.numWord):
                self.TFIDF[i][j] = TF[i][j] * IDF[j]

    def filterDF(self,percentLowerBound=0.05,percentUpperBound=0.95):
        maxDF = -1
        for i in range(self.numWord):
            if maxDF < self.DF[i]:
                maxDF = self.DF[i]

        lstSelectedWordIdx = []
        for i in range(self.numWord):
            if self.DF[i] / maxDF >= percentLowerBound  and \
                self.DF[i] / maxDF <= percentUpperBound:
                lstSelectedWordIdx.append(i)

        filteredCorpusMatrix, filteredColHeading = self.reduceBOWbySelection(lstSelectedWordIdx)
        return MatrixDataset(filteredCorpusMatrix, self.rowHeading, filteredColHeading)

    def filterShortWord(self,wordMinimumLength):
        lstSelectedWordIdx = []
        for i in range(self.numWord):
            if len(self.colHeading[i]) >= wordMinimumLength:
                lstSelectedWordIdx.append(i)

        filteredCorpusMatrix, filteredColHeading = self.reduceBOWbySelection(lstSelectedWordIdx)
        return MatrixDataset(filteredCorpusMatrix, self.rowHeading, filteredColHeading)

    def filterAggregatedTFIDF(self,percentLowerBound=0.05,percentUpperBound=0.95):
        aggregatedTFIDF = []
        for i in range(self.numWord):
            summation = 0
            for j in range(self.numDoc):
                summation = summation + self.TFIDF[j][i]
            aggregatedTFIDF.append(summation)

        aggregatedMaxTFIDF = -1
        for i in range(self.numWord):
            if aggregatedMaxTFIDF < aggregatedTFIDF[i]:
                aggregatedMaxTFIDF = aggregatedTFIDF[i]

        lstSelectedWordIdx = []
        for i in range(self.numWord):
            if aggregatedTFIDF[i] / aggregatedMaxTFIDF >= percentLowerBound  and \
                                    aggregatedTFIDF[i] / aggregatedMaxTFIDF <= percentUpperBound:
                lstSelectedWordIdx.append(i)

        filteredCorpusMatrix, filteredColHeading = self.reduceBOWbySelection(lstSelectedWordIdx)
        return MatrixDataset(filteredCorpusMatrix, self.rowHeading, filteredColHeading,isSparse=True)

    def reduceBOWbySelection(self,lstSelectedWordIdx):
        filteredColHeading = []
        filteredCorpusMatrix = []

        for i in range(len(lstSelectedWordIdx)):
            idx = lstSelectedWordIdx[i]
            filteredColHeading.append(self.colHeading[idx])

        for i in range(self.numDoc):
            filterdRow = {}
            lstIdxExistingWordInDoc = self.matrixData.getNonZeroVectorElementIdx(i)
            for j in range(len(lstSelectedWordIdx)):
                idx = lstSelectedWordIdx[j]
                if idx in lstIdxExistingWordInDoc:
                    filterdRow[idx] = self.matrixData.getMatrixValue(i,idx)
            filteredCorpusMatrix.append(filterdRow)

        return filteredCorpusMatrix, filteredColHeading


if __name__ == "__main__":
    reader = BagOfWordReader('../../TestDataset/SentimentDataset-bow.csv',\
                             strColHeadingFileName='../../TestDataset/SentimentDataset-words.csv')
    matrixData = reader.getResult()

    filter = BagOfWordFilter(matrixData)
    filter.filterDF()
    filter.filterAggregatedTFIDF()



