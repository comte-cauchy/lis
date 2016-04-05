import csv
import datetime
import numpy as np


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
fileName = 'train.csv'
nOutput = 1
nInput = 15
with open(fileName,'r') as inputFile:
    reader = csv.reader(inputFile)
    next(reader)
    for row in reader:
        trainid.append(int(row[0]))
        trainy.append([float(el) for el in row[1:nOutput+1]])
        trainx.append([float(el) for el in row[nOutput+1:nInput+nOutput+1]])

        
print("Reading test data...")
fileName = 'test.csv'
with open(fileName,'r') as inputFile:
    reader = csv.reader(inputFile)
    next(reader)
    for row in reader:
        testid.append(int(row[0]))
        testx.append([float(el) for el in row[1:nInput+1]])

trainx = np.array(trainx)
trainy = np.array(trainy)
testx = np.array(testx)
######################################
#Learning
######################################

### Config section
doOutput = True
import k_NN_CV as selectedMethod

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