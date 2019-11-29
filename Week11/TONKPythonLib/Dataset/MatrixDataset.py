

class MatrixDataset:

    __matrix = []
    __rowHeading = []
    __colHeading = []
    isSparse = False

    def __init__(self, matrix, rowHeading, colHeading, isSparse = False):
        self.__matrix = matrix
        self.__rowHeading = rowHeading
        self.__colHeading = colHeading
        self.isSparse = isSparse


    def getMatrixValue(self,idxRow,idxCol):
        return self.__matrix[idxRow][idxCol]

    def getNonZeroVectorElementIdx(self,idxRow):
        if self.isSparse == True:
            return self.__matrix[idxRow].keys()
        else:
            ret = []
            for i in range(len(self.__matrix[idxRow])):
                if self.__matrix[idxRow][i] != 0:
                    ret.append(i)
            return ret

    def getRowHeading(self):
        return self.__rowHeading

    def getColHeading(self):
        return self.__colHeading
