import csv
import lisData
import numpy as np
listli = ["1","123.45","-1e6"]
listfl = np.array(map(np.float, listli))
print(listfl)
data = lisData.LisData('train.csv',10,1)

for s in data.data:
    print(s.id,',',', '.join(map(str,s.x)),', ',np.mean(s.x))
with open('output.csv', 'w') as outFile:
    outFile.write('Id,y\n')
    for s in data.data:
        outFile.write("%d,%s\n"%(s.id,format(np.mean(s.x),'.16g')))