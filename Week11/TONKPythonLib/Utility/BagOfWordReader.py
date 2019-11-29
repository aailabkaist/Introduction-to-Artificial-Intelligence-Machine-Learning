import csv
from TONKPythonLib.Dataset.MatrixDataset import MatrixDataset

class BagOfWordReader:

    corpusMatrix = []
    rowHeading = []
    colHeading = []

    def __init__(self,strMatrixFileName,strColHeadingFileName,strRowHeadingFileName=''):
        self.readBagOfWord(strMatrixFileName)

        if strRowHeadingFileName == '':
            numRow = len(self.corpusMatrix)
            for i in range(numRow):
                self.rowHeading.append( 'Doc_'+str(i) )
        else:
            self.rowHeading = self.readHeading(strRowHeadingFileName)

        if strColHeadingFileName == '':
            numCol = len(self.corpusMatrix[0])
            for i in range(numCol):
                self.colHeading.append( 'Word_'+str(i) )
        else:
            self.colHeading = self.readHeading(strColHeadingFileName)

    def getResult(self):
        return MatrixDataset(self.corpusMatrix, self.rowHeading, self.colHeading)

    def readBagOfWord(self,strFileName):
        file = open(strFileName,'r')
        try:
            reader = csv.reader(file)
            for row in reader:
                rowInCorpus = []
                for col in row:
                    rowInCorpus.append(int(col))
                self.corpusMatrix.append(rowInCorpus)
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