from TONKPythonLib.Dataset.MatrixDataset import MatrixDataset
from numpy import zeros
class SparseBagOfWordReader:

    corpusMatrix = []
    rowHeading = []
    colHeading = []

    def __init__(self,strMatrixFileName,strColHeadingFileName,strRowHeadingFileName=''):
        if strColHeadingFileName == '':
            numCol = len(self.corpusMatrix[0])
            for i in range(numCol):
                self.colHeading.append( 'Word_'+str(i) )
        else:
            self.colHeading = self.readHeading(strColHeadingFileName)

        self.readBagOfWord(strMatrixFileName)

        if strRowHeadingFileName == '':
            numRow = len(self.corpusMatrix)
            for i in range(numRow):
                self.rowHeading.append( 'Doc_'+str(i) )
        else:
            self.rowHeading = self.readHeading(strRowHeadingFileName)


    def getResult(self):
        return MatrixDataset(self.corpusMatrix, self.rowHeading, self.colHeading, isSparse=True)

    def readBagOfWord(self,strFileName):
        file = open(strFileName,'r')
        try:
            while True:
                line = file.readline().strip()
                doc = line.split()
                if len(doc) <= 1:
                    file.close()
                    return
                docVector = {}
                for i in range(1,len(doc)):
                    token = doc[i].split(":")
                    idx = int(token[0])
                    count = int(token[1])
                    docVector[idx] = count
                self.corpusMatrix.append(docVector)
        finally:
            file.close()

    def readHeading(self,strFileName):
        file = open(strFileName, 'r')
        try:
            heading = []
            while True:
                line = file.readline().strip()
                if not line:
                    break
                heading.append(line)
        finally:
            file.close()
            return heading

if __name__ == "__main__":
    reader = SparseBagOfWordReader('../../TestDataset/ap.dat', \
                                   strColHeadingFileName='../../TestDataset/vocab.txt'
                             )
    matrixData = reader.getResult()

    print matrixData.getColHeading()
    print matrixData.getRowHeading()
    print matrixData.getMatrix()
