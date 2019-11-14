from sklearn.datasets import load_boston
from sklearn.linear_model import (LogisticRegression, Ridge, Lasso)                                
from stability_selection import RandomizedLogisticRegression
from sklearn.feature_selection import RFE, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from minepy import MINE
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

feature_sort_name = 'feature_sort.csv'

X_train = np.load('./data/X_train_XGB_tor_VPN_before_smote_all_class.npy',allow_pickle=True)
y_train = np.load('./data/y_train_XGB_tor_VPN_before_smote_all_class.npy',allow_pickle=True)

names = pd.read_csv('./prepro/ColSample.csv').columns.ravel()
#x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.3, random_state=21)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=21)
ranks = {}

def _get_feature_importances(estimator, norm_order=1):
     """Retrieve or aggregate feature importances from estimator"""
     importances = getattr(estimator, "feature_importances_", None)

     coef_ = getattr(estimator, "coef_", None)
     if importances is None and coef_ is not None:
          if estimator.coef_.ndim == 1:
               importances = np.abs(coef_)

          else:
               importances = np.linalg.norm(coef_, axis=0,
                                         ord=norm_order)

     elif importances is None:
          raise ValueError(
             "The underlying estimator %s has no `coef_` or "
            "`feature_importances_` attribute. Either pass a fitted estimator"
            " to SelectFromModel or call fit before calling transform."
            % estimator.__class__.__name__)

     return importances

def rank_to_dict(ranks, names, order=1):
     #scores = _get_feature_importances(estimator, order)
        
     minmax = MinMaxScaler()
     ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
     ranks = map(lambda x: np.round(x, 2), ranks)
     return dict(zip(names, ranks ))

from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
##iris = load_iris()
##X, y = iris.data, iris.target
##X.shape
##lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
##model = SelectFromModel(lsvc, prefit=True)
##X_new = model.transform(X)

f, pval  = f_classif(x_train, y_train)
ranks["Corr."] = rank_to_dict(f, names)

lg_l1 = LogisticRegression(penalty='l1', multi_class='ovr', solver='liblinear', max_iter=10000)
lg_l1.fit(x_train, y_train)
print(lg_l1.score(x_test, y_test))
print(lg_l1.predict(x_val))
lg_l1.coef_ = _get_feature_importances(lg_l1)
ranks["Logistic classify L1"] = rank_to_dict(lg_l1.coef_, names)

lg_l2 = LogisticRegression(penalty='l2', multi_class='ovr', solver='liblinear',max_iter=10000)
lg_l2.fit(x_train, y_train)
print(lg_l2.score(x_test, y_test))
lg_l2.coef_ = _get_feature_importances(lg_l2)
ranks["Logistic classify L2"] = rank_to_dict(lg_l2.coef_, names)
print('1')
rlg = RandomizedLogisticRegression(max_iter=10000)
rlg.fit(x_train, y_train)
rlg.coef_= _get_feature_importances(rlg)
ranks["Stability"] = rank_to_dict(rlg.coef_, names)
print('2')
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
ranks["RF"] = rank_to_dict(rf.feature_importances_, names)
print('5')

mine = MINE()
mic_scores = []
for i in range(x_train.shape[1]):
     mine.compute_score(x_train[:,i], y_train)
     m = mine.mic()
     mic_scores.append(m)

ranks["MIC"] = rank_to_dict(mic_scores, names)
print('6')

##stop the search when 5 features are left (they will get equal scores)
lsvc = LinearSVC(C=1, penalty="l2", dual=False, max_iter = 10000).fit(x_train, y_train)
##model = SelectFromModel(lsvc, prefit=True)
##X_new = model.transform(x_test)
print('3')
rfe = RFE(rf, n_features_to_select=1, step=1)
rfe.fit(x_train, y_train)
ranks["RFE"] = rank_to_dict(list(map(float, rfe.ranking_)), names, order=-1)
r = {}
for name in names:
     r[name] = round(np.mean([ranks[method][name] 
                             for method in ranks.keys()]), 2)

methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")

feature_sort = pd.DataFrame(index = names, columns= ranks.keys(), data=ranks)
feature_sort.to_csv('feature_sort_new.csv')