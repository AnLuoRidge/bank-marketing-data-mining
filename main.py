#%% modules
import Preprocess_Train, Preprocess_Test, Draw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from tpot import TPOTClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

#%% preprocess
train_location = '/TrainingSet.csv'
test_location = '/TestingSet.csv'
df_preprocessed = Preprocess_Train.preprocess(train_location)
df_test = Preprocess_Test.preprocess(test_location)
# For testing
data = df_preprocessed.drop(['Final_Y'], axis=1)
label = df_preprocessed.Final_Y
# For training
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2) # random_state=34

best_features = [True, True, True, True, False, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, True]

#%% Resampling
smote = SMOTE()
x_train, y_train = smote.fit_resample(x_train, y_train)
x_train = pd.DataFrame(x_train)
y_train = pd.DataFrame(y_train)

#%% Feature selection
def feature_selection(classifier):
    start_num = 1
    end_num = 36
    temp_list = []

    for step in range(start_num, end_num):
        selector = RFE(classifier, step, step=1)
        y_pred = selector.fit(x_train, y_train).predict(x_test)
        score = f1_score(y_test, y_pred, average='binary')
        temp_list.append((score, step, selector.support_))
    best = max(temp_list, key=lambda x: x[0])
    print(best) # the best num of features and feature details
    return  best[2]
best_features = feature_selection(DecisionTreeClassifier()) # DC: 9 features

#%% parameter tuning - Decision Tree
parameters = [{
               'max_depth': range(1, 30),
               'min_samples_split': range(2, 30),
               'criterion': ('entropy', 'gini'),
               'splitter': ('random', 'best'),
               'min_samples_leaf': range(1, 30),
               'max_leaf_nodes': range(2, 30)
               }]
clf = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5, scoring='f1', verbose=0, n_jobs=-1)
y_pred = clf.fit(x_train.loc[: , best_features], y_train).predict(x_test.loc[: , best_features])
print(clf.cv_results_['mean_test_score'])
print(clf.best_params_)

Draw.plot_confusion_matrix(y_test, y_pred, title='Decision Tree')
plt.show()

#%% parameter tuning - K-nearest neighbour
def KNN():
    start_num = 1
    end_num = 100
    temp_list = []

    for step in range(start_num, end_num):
        knn_model = KNeighborsClassifier(n_neighbors=step)
        y_pred = knn_model.fit(x_train, y_train).predict(x_test)
        score = f1_score(y_test, y_pred, average='binary')
        temp_list.append(((score, step), step))

    print(max(temp_list, key=lambda x: x[0]))

#%% Parameter tuning - GB
parameters = [{
               'n_estimators': range(1, 100),
               'learning_rate': np.arange(0.1, 0.5, 0.01),
               'max_depth': range(1, 20),
               'min_samples_split': range(2, 10),
               'max_leaf_nodes': range(1, 20),
               'min_samples_leaf': range(1, 20)
               }]
clf = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5, scoring='f1', verbose=0, n_jobs=-1)
y_pred = clf.fit(x_train.loc[: , best_features], y_train).predict(x_test.loc[: , best_features])
print(clf.cv_results_['mean_test_score'])
print(clf.best_params_)

Draw.plot_confusion_matrix(y_test, y_pred, title='Decision Tree')
plt.show()

#%% Parameter tuning - MLP
def MLP():
    start_num = 1
    end_num = 20
    temp_list = []
# 2 layers are enough
    for first in range(start_num, end_num):
        for second in range(start_num, end_num):
            for third in range(start_num, end_num):
                mlp = MLPClassifier((first, second, third), max_iter=500)
                y_pred = mlp.fit(x_train.loc[:, best_features], y_train).predict(x_test.loc[:, best_features])
                score = mlp.score(x_test.loc[:, best_features], y_test)
                temp_list.append((score, first, second))
    best = max(temp_list, key=lambda x: x[0])
    print(best)
MLP()

#%% GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=113, learning_rate=0.2, max_depth=3)
clf.fit(x_train.loc[:, best_features], y_train)
y_pred = clf.predict(x_test.loc[:, best_features])
print(f1_score(y_test, y_pred, average='binary'))
Draw.plot_confusion_matrix(y_test, y_pred, title='Gradient Boosting')
plt.show()
clf.score(x_test.loc[:, best_features], y_test)

#%% Random Forest
clf = RandomForestClassifier(n_estimators=70, criterion='entropy', max_depth=9, min_samples_split=6, min_samples_leaf=1, max_features=4, max_leaf_nodes=None)
clf.fit(x_train.loc[:, best_features], y_train)
y_pred = clf.predict(x_test.loc[:, best_features])
print(f1_score(y_test, y_pred, average='binary'))
Draw.plot_confusion_matrix(y_test, y_pred, title='Random forest')
plt.show()

clf.score(x_test.loc[:, best_features], y_test)
Draw.PR_curve(y_test, y_pred)

#%%
def tpot():
    df = Preprocess_Train.preprocess(train_location)
    data = df.drop(['Final_Y'], axis=1)
    label = df.Final_Y
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=34)
    tpot = TPOTClassifier(scoring='f1',
                          max_time_mins=300,
                          n_jobs=-1,
                          verbosity=2,
                          cv=5)
    tpot.fit(data, label)
    tpot.score(x_test, y_test)
    # LinearSVC(C=25.0, dual=True, loss="squared_hinge", penalty="l2", tol=0.1)

#%%
def Output():
    rf_model = RandomForestClassifier(n_estimators=70)
    rf_model.fit(data, label)
    result = rf_model.predict(df_test)
    result = pd.Series(result)
    result_csv = 'submit.csv'
    df_submit = pd.DataFrame()
    df_submit['Final_Y'] = result
    df_submit.to_csv(result_csv)
