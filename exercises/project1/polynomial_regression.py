import numpy as np
import sklearn.linear_model as sklin
import sklearn.metrics as skmet
import sklearn.cross_validation as skcv
import sklearn.preprocessing as skpre
import matplotlib.pyplot as plt
import util

def toString():
    return 'polynomial_ridge_regression'

def predict(trainx, trainy, testx):
    scorer = skmet.make_scorer(util.score)
    #best_score = np.Inf
    #for degree in range(5):
        #polynomial_features = skpre.PolynomialFeatures(degree)
        #transx = polynomial_features.fit_transform(trainx)
        #for alphaLoc in np.linspace(100*(degree-1),100*(degree+1),num=101):
            #regressor = sklin.Ridge(alpha=alphaLoc)
            #scores = skcv.cross_val_score(regressor, transx, trainy, scoring=scorer, cv=5)
            #print('C-V score for degree=',degree,'and alpha=',alphaLoc,'is', np.mean(scores), '+/-',np.std(scores))

            #if np.mean(scores)<best_score:
                #best_score = np.mean(scores)
                #best_std = np.std(scores)
                #best_degree = degree
                #best_alpha = alphaLoc
        #print('C-V score for degree=',degree,'and alpha=',best_alpha,'is', best_score, '+/-',best_std)
        
        ##print(regressor.alphas)       
        ##print('    alpha=',regressor.alpha_,'was used.') 
    ##clf = sklin.RidgeCV( scoring=scorer, alphas=np.linspace(0, 5), normalize=False, cv=10)
    
    
    #polynomial_features = skpre.PolynomialFeatures(best_degree)
    polynomial_features = skpre.PolynomialFeatures(3)
    transx = polynomial_features.fit_transform(trainx) 
    transx_test = polynomial_features.fit_transform(testx)
    
    #regressor = sklin.Ridge(alpha=best_alpha)
    regressor = sklin.Ridge(alpha=260)
    regressor.fit(transx,trainy)
    testy = regressor.predict(transx_test)
    print("On the training set, RMSE  of the model is ",util.score(trainy,regressor.predict(transx)))
 #   print("The used degree is ", best_degree)
    return testy