import datetime
import numpy as np
import pandas as pd
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")


trainid = []
trainx = []
trainy = []

testid = []
testx = []
testy = []

######################################
#Read in Files
######################################
print("Reading training data...")
fileName = 'train.h5'
nOutput = 1
nInput = 100
train = pd.read_hdf(fileName, "train")
#with open(fileName,'r') as inputFile:
    #reader = csv.reader(inputFile)
    #next(reader)
    #for row in reader:
        #trainid.append(int(row[0]))
        #trainy.append([float(el) for el in row[1:nOutput+1]])
        #trainx.append([float(el) for el in row[nOutput+1:nInput+nOutput+1]])

        
print("Reading test data...")
fileName = 'test.h5'
test = pd.read_hdf(fileName, "test")
#with open(fileName,'r') as inputFile:
    #reader = csv.reader(inputFile)
    #next(reader)
    #for row in reader:
        #testid.append(int(row[0]))
        #testx.append([float(el) for el in row[1:nInput+1]])

trainx = train.as_matrix()[:,1:]
trainy = train.as_matrix()[:,0]

testx = test.as_matrix()
testid = list(test.index)
######################################
#Learning
######################################

### Config section
doOutput = True
import rbf_kernel as selectedMethod

### run learning
print("Begin learning section with method", selectedMethod.toString(),"...")
testy = selectedMethod.predict(trainx,trainy,testx)

print("End of learning section...")

######################################
#Write Results
######################################
if doOutput:
    now =datetime.datetime.now()
    outFileName = 'output_{0}_{1}.csv'.format(now.strftime('%Y%m%d%H%M'),selectedMethod.toString())
    print("Writing Result to ",outFileName,"...")
    with open(outFileName, 'w') as outFile:
        outFile.write('Id,y\n')
        for idx,testyelem in enumerate(testy):
            #outFile.write("%d,%s\n"%(idx+900,format(testyelem,'.30f')))    
            outFile.write("%d,%d\n"%(testid[idx],testyelem))
            
    print("Done.")