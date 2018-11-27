"""
working on sklearn's inbuilt BreastCancer dataset using SVC to classify tumor .

this code has 3 approaches step by step
1) naive solution -> naive_solution
    [[ 0 47]
    [ 0 67]]  it is not good needs improvement =>>
2) scaling the features -> apply_normalization
    scaling is acheived -> manually & by sklearn's MinMaxScalar
    # [[46  1]
    #  [6 61]]
3) tuning SVC params with scaled features -> with_tuning
    [[45  3]
    [0 66]]   # better than only scaled solution.
    svc has c parameter =>  trade of b/w correct classification & having smooth decision boundary.
    c (small) -> on wrong classification less penalty
    c (large) -> large penalty  =>> causes over fitting
    gamma --> about decision boundary
    large -> focus on points near decision boundary
    small -> far reach -> generalized solution .
    kernel-> linear, poly, rbf
    either manually best params can be found or Grid Search be used.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# helper function to plot some data using sns
def data_visualization():
    # pair wise relation hue differentiate the true/false
    sns.pairplot(df, hue='target', vars=['mean radius', 'mean texture', 'mean area'])
    plt.show()
    sns.countplot(df['target'])
    plt.show()
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=True)
    plt.show()


def naive_solution(df) :
    # make a classifier
    classifier = SVC()
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    # [[ 0 47]
    #  [ 0 67]]  it is not good needs improvement =>>
    sns.heatmap(cm,annot=True)

def apply_normalization(df):
    # make a classifier
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    classifier = SVC()
    X = df.drop('target', axis=1)
    y = df['target']
    #scale X .
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #scale X_train , X_test
    # scaling = ( x- xmin )(xmax-xmin )
    min_train = X_train.min()
    range_train = (X_train-min_train).max()
    scale_x_train = (X_train -min_train)/range_train

    min_test = X_test.min()
    range_test = (X_test-min_test).max()
    scale_x_test = (X_test-min_test)/range_test

    # sns.scatterplot(x= X_train['mean area'], y =X_train['mean smoothness'] , hue= y_train)
    # plt.show()
    # sns.scatterplot(x= scale_x_train['mean area'], y =scale_x_train['mean smoothness'] , hue= y_train)
    # plt.show()
    # plt.show()

    classifier.fit(scale_x_train, y_train)

    y_predict = classifier.predict(scale_x_test)

    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    # [[46  1]
    #  [6 61]]   ( only 7 wrong classifications , better than naive solution )
    sns.heatmap(cm,annot=True)
    plt.show()

def with_tuning(df):
    # make a classifier
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix ,classification_report
    from sklearn.model_selection import train_test_split
    param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'kernel':['linear','poly','rbf']}
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import MinMaxScaler
    min_max_scalar = MinMaxScaler(feature_range=(0,1))
    classifier = GridSearchCV(SVC(), param_grid,refit=True,verbose=4)
    X = df.drop('target', axis=1)
    y = df['target']
    X = min_max_scalar.fit_transform(X)
    #scale X .
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    #scale X_train , X_test
    # scaling = ( x- xmin )(xmax-xmin )
    classifier.fit(X_train, y_train)
    print("best params",classifier.best_params_)
    y_predict = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    # [[45  3]
    #  [0 66]]   # better than only scaled solution.
    sns.heatmap(cm,annot=True)
    plt.show()
    report=classification_report(y_test,y_predict)
    print(report)



cancer = load_breast_cancer()
#dict_keys(['feature_names', 'filename', 'target', 'data', 'target_names', 'DESCR'])
#make 2d array from data and target .
data = np.c_[cancer['data'],cancer['target']]
cols = cols = np.append(cancer['feature_names'],['target'])
#make a data frame from pandas
df = pd.DataFrame(data, columns= cols)

naive_solution(df)
apply_normalization(df)
with_tuning(df)