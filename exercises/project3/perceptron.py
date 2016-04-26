import sklearn.linear_model as sklin
import sklearn.cross_validation as skcv
import sklearn.preprocessing as skpre
import numpy as np

def toString():
    return 'perceptron'

def predict(trainx,trainy,testx):
    np.set_printoptions(precision=4)
    
    trainy = np.reshape(trainy,[len(trainy),])

    nTrain = len(trainy)
    
    best_pen = ''
    best_deg = 0
    best_alpha = 0
    best_score = 0

    
    for pen in ['l1','l2','elasticnet']:    
        for deg in range(1,6):
            polynomial_features = skpre.PolynomialFeatures(deg)
            transx = polynomial_features.fit_transform(trainx) 
            if pen == 'l1':
                for alp in np.linspace(0,3,10):
                    print('Up next: degree =',deg,'penalty =',pen,'alpha =',alp)
                    clf = sklin.Perceptron(penalty=pen,alpha=alp)
                    scores = skcv.cross_val_score(clf, transx,trainy,cv = 10)
                    print('...Score was',np.mean(scores),'+/-',np.std(scores))
                    if(np.mean(scores)>best_score):
                        best_score = np.mean(scores)
                        best_pen = pen
                        best_deg = deg
                        best_alpha = alp
            else:
                print('Up next: degree =',deg,'penalty =',pen)
                clf = sklin.Perceptron(penalty=pen)
                scores = skcv.cross_val_score(clf, transx,trainy,cv = 10)
                print('...Score was',np.mean(scores),'+/-',np.std(scores))
                if(np.mean(scores)>best_score):
                    best_score = np.mean(scores)
                    best_pen = pen
                    best_deg = deg
                    best_alpha = 0                
                
    print('Best CV result: degree =',best_deg,'penalty =',best_pen,'alpha =',best_alpha, 'score =',best_score)                

    polynomial_features = skpre.PolynomialFeatures(best_deg)
    transx = polynomial_features.fit_transform(trainx) 
    transx_test = polynomial_features.fit_transform(testx)     
    clf = sklin.Perceptron(penalty=best_pen,alpha=best_alpha)
    clf.fit(transx, trainy)
    return clf.predict(transx_test)
            
                
    