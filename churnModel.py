import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.linear_model import LogisticRegression
'''
logistic 
[[1651 1515]
 [ 566 1668]]
 
 svm
 [[1737 1429]
 [ 582 1652]]
 
 tuned svm
 [[1677 1489]
 [ 721 1513]]
 
 decision tree
 [[1855 1311]
 [ 741 1493]]
 
 random forest
 [[2150 1016]
 [ 783 1451]]
 
 naive bayes
 
 [[1393 1773]
 [ 528 1706]]
 
 knn
 [[1777 1389]
 [ 887 1347]]
 '''
def make_report_and_confustion_mat(y_true, y_predict):
    cm = confusion_matrix(y_test, y_predict)
    report = classification_report(y_test, y_predict)
    return cm, report


def logisitic_regression(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression(random_state=0, penalty='l1')
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    return "logi", make_report_and_confustion_mat(y_test,y_predict)


def svm(X_train , X_test , y_train , y_test):
    from sklearn.svm import SVC
    classifier = SVC()
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    return "svm", make_report_and_confustion_mat(y_test,y_predict)

def tuned_svm(X_train, X_test , y_train , y_test ):
    param_grid = {'C':[1,10,100],'gamma':[0.1,0.01,0.001]}
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    classifier = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print("best params", classifier.best_params_)
    return "tuned svm", make_report_and_confustion_mat(y_test,y_predict)


def decision_tree(X_train, X_test , y_train , y_test ):
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    return "decision tree ", make_report_and_confustion_mat(y_test,y_predict)

def knn(X_train, X_test , y_train , y_test):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    return "knn", make_report_and_confustion_mat(y_test,y_predict)

def random_forest(X_train, X_test, y_train, y_test) :
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    return "random forest", make_report_and_confustion_mat(y_test,y_predict)


def naive_bayes(X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    return "naive_bayes", make_report_and_confustion_mat(y_test,y_predict)


def balance_training_set(X_train,y_train) :
    type_1 = y_train[y_train.values==1].index
    type_0 = y_train[y_train.values==0].index
    if len(type_1) > len(type_0) :
        max = type_1
        min = type_0
    else :
        min = type_1
        max = type_0

    print('class 1',len(type_1),'class 0', len(type_0),'going with ', len(min),'\n')
    import random
    random.seed(0)
    max = np.random.choice(max, size=len(min))
    min = np.asarray(min)
    new_idx = np.concatenate((min,max))
    X_train = X_train.loc[new_idx,]
    y_train = y_train[new_idx]
    return X_train , y_train


def apply_standard_scalar(X_train , X_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train2 = pd.DataFrame(sc.fit_transform(X_train))
    X_test2 = pd.DataFrame(sc.transform(X_test))
    X_train2.columns = X_train.columns
    X_test2.columns = X_test.columns
    X_train2.index = X_train.index
    X_test2.index = X_test.index
    return X_train2,X_test2


data = pd.read_csv('churn_data_processed.csv')
# print(data.describe())
# print(data.isnull().sum().max())

user = data['user']
data = data.drop(columns=['user'])

#one hot encoding for categorical variables
data = pd.get_dummies(data)
# avoid dummy variable trap => housing ( na , o , r => if o and r 0 then its means na => correlated fields )
data = data.drop(columns=['housing_na','payment_type_na','zodiac_sign_na'])

X = data.drop(columns=['churn'])
y = data['churn']
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X ,y ,test_size=0.2,random_state=0 )

# standard scalar .
# 0    12656
# 1     8940
# it is possible that this model is trained more on 0 output which might have impact on its accuracy.

# way to handle is keep same number of zeros and 1's
X_train, y_train = balance_training_set(X_train,y_train)
X_train ,X_test = apply_standard_scalar(X_train,X_test)

all_classifiers = [logisitic_regression, svm , tuned_svm , decision_tree,random_forest ,naive_bayes , knn]
file = open('churn_model','w')
for fun in all_classifiers :
    name, cm_report = fun(X_train,X_test,y_train,y_test)
    file.write("\n--------"+name+"-----\n")
    file.write(str(cm_report[0])+"\n")
    file.write(str(cm_report[1]))

file.close()

from sklearn.feature_selection import RFE
classifier = LogisticRegression()
rfe = RFE(classifier , 4)
rfe = rfe.fit(X_train,y_train)
classifier = LogisticRegression(random_state=0, penalty='l1')
classifier.fit(X_train[X_train.columns[rfe.support_]], y_train)
y_predict = classifier.predict(X_test[X_test.columns[rfe.support_]])
print(classification_report(y_test,y_predict))
print(classifier.coef_)
print(classifier.classes_)