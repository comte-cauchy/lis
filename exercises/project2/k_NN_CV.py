import sklearn.neighbors as skneigh
import sklearn.cross_validation as skcv
import numpy as np

def toString():
    return 'k-NN'

def predict(trainx,trainy,testx):
    trainy = np.reshape(trainy,[len(trainy),])
    nTrain = len(trainy)
    best_score = 0
    best_weights = ''
    best_k = 0
    for w in ['uniform','distance']:
        for k in range(1,int(0.1*nTrain)):
            print('Up next: weights =',w,'k =',k)
            neigh = skneigh.KNeighborsClassifier(n_neighbors=k, weights=w)
            scores = skcv.cross_val_score(neigh, trainx,trainy,cv = 10)
            print('...Score was',np.mean(scores),'+/-',np.std(scores))
            if(np.mean(scores)>best_score):
                best_score = np.mean(scores)
                best_k = k
                best_weights = w
                
    print('Best CV result: weights =',best_weights,'k =',best_k, 'score =',best_score)                
    neigh = skneigh.KNeighborsClassifier(n_neighbors=best_k, weights = best_weights)
    neigh.fit(trainx, trainy)
    return neigh.predict(testx)
            
                
    