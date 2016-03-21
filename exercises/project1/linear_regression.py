import sklearn.linear_model as sklin
import util

def toString():
    return 'linear_regression'

def predict(trainx, trainy, testx):
    clf.fit(trainx,trainy)
    testy = clf.predict(testx)
    print("On the training set, RMSE  of the model is ",util.score(trainy,clf.predict(trainx)))
    return testy