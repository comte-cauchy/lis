import csv
import numpy as np
class LisSample:
    def __init__(self,id,x,y):
        self.id = id
        self.x = x
        self.y = y

class LisData:
    def __init__(self, fileName,nInput,nOutput):
        self.data = []
        with open(fileName,'r') as inputFile:
            leReader = csv.reader(inputFile)
            next(leReader)
            for row in leReader:
                id = int(row[0])
                y = np.array(row[1:nOutput+1]).astype(np.float)
                x = np.array(row[nOutput+1:nInput+nOutput+1]).astype(np.float)
                self.data.append(LisSample(id, x, y))
                
                    
            
