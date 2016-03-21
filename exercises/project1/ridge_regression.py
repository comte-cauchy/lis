import numpy as np
import sklearn.linear_model as sklin
import sklearn.metrics as skmet
import sklearn.cross_validation as skcv
import util

def toString():
    return 'ridge_regression'

def predict(trainx, trainy, testx):
    scorer = skmet.make_scorer(util.score)
    best_score = np.Inf
    for alphaLoc in np.linspace(0,100,101):
        regressor = sklin.Ridge(alpha=alphaLoc,fit_intercept=True)
        scores = skcv.cross_val_score(regressor, trainx, trainy, scoring=scorer, cv=10)
        if np.mean(scores)<best_score:
            best_score = np.mean(scores)
            best_alpha = alphaLoc;
        print('C-V score for alpha=',alphaLoc,'is', np.mean(scores), '+/-', np.std(scores))
    #clf = sklin.RidgeCV( scoring=scorer, alphas=np.linspace(0, 5), normalize=False, cv=10)
    regressor = sklin.Ridge(alpha=best_alpha)
    regressor.fit(trainx,trainy)
    testy = regressor.predict(testx)
    print("On the training set, RMSE  of the model is ",util.score(trainy,regressor.predict(trainx)))
    print("The used alpha is ", regressor.alpha)
    return testy