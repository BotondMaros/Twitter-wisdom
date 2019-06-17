from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from featureExtract import *

def main():
    # Path to the training data file
    x, y, vec = featureBuilder("../data/train_data.csv")
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    model.fit(x, y)
    
    # Path to the testing data file, must use the dictionary vec from training
    x_test, y_test, v = featureBuilder("../twitter-wisdom-master/data/test_data.csv", vec)
    y_pred = model.predict(x_test)
    print(metrics.classification_report(y_pred, y_test))
    
def mainProba():
    x, y, vec = featureBuilder("../data/train_data.csv")
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    model.fit(x, y)
    
    x_test, y_test, v = featureBuilder("../twitter-wisdom-master/data/test_data.csv", vec)
    y_pred = model.predict_proba(x_test)
    y_p = []
    y_test1 = []
    for i in y_pred:
    if i[2] >= threshold():
        y_p.append(1)
    elif i[0] >= threshold():
        y_p.append(-1)
    else:
        y_p.append(0)
    for i in y_test:
        if i == -1:
            y_test1.app
    print(sum(y_test))
    print(metrics.classification_report(y_p, y_test))
