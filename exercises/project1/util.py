import sklearn.metrics as skmet

def score(y1, y2):
    return skmet.mean_squared_error(y1,y2)**0.5