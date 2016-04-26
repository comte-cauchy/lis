import sklearn.svm as sksvm
import sklearn.cross_validation as skcv
import sklearn.preprocessing as skpre
import numpy as np

def toString():
    return 'rbf_kernel'

def predict(trainx,trainy,testx):
    trainx = trainx[:,:]
    trainy = trainy[:]
    np.set_printoptions(precision=4)
    
    trainy = np.reshape(trainy,[len(trainy),])

    nTrain = len(trainy)
    mi = -1
    
    best_deg = 0
    best_c = 0
    best_gamma = 0
    best_score = 0
    ker = 'rbf'
    for deg in range(1,2):
        polynomial_features = skpre.PolynomialFeatures(deg)
        transx = polynomial_features.fit_transform(trainx) 
        for c in np.logspace(0,.5,9):#np.logspace(-1,1,9):
            for gam in np.logspace(-1,0,9):#np.logspace(-np.log10(nTrain)-1,0,9):
                print('Up next: degree =',deg,'c =',c, 'gamma=',gam)
                clf = sksvm.SVC(C=c,gamma=gam,kernel=ker,max_iter = mi)
                scores = skcv.cross_val_score(clf, transx,trainy,cv = 3, n_jobs=-1)
                print('...Score was',np.mean(scores),'+/-',np.std(scores))
                if(np.mean(scores)>best_score):
                    best_score = np.mean(scores)
                    best_deg = deg
                    best_c = c
                    best_gamma = gam
                
    print('Best CV result: degree =',best_deg,'c =',best_c, 'gamma = ',best_gamma,'score =',best_score)                

    polynomial_features = skpre.PolynomialFeatures(best_deg)
    transx = polynomial_features.fit_transform(trainx) 
    transx_test = polynomial_features.fit_transform(testx)     
    clf = sksvm.SVC(C=best_c,gamma=best_gamma,kernel=ker)

    clf.fit(transx, trainy)
    return clf.predict(transx_test)
    
                
    