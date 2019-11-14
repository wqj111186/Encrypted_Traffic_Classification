import os
import sys
sys.path.insert(0, os.environ['HOME'] + '/BotnetDetectionThesis/')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import svm, tree

import numpy as np
import get_normalize_data
import config as c
from logging import getLogger
from model_botnet import ML_Model
import random

def select_models(models, models_name):
    return [m for m in models if m.name in models_name]


def final_train(models, X_train, y_train, X_test, y_test, set_name):
    with open(c.training_output_file, 'a') as f:
        f.write("Features set : " + set_name + "\n")
        for model in models:
            model.train(X_train, y_train)
            model.predict(X_test, y_test)
            #model.compute_metrics(y_test)
            #logger.debug(model.get_printable_metrics())
            #with open(c.training_output_file, 'a') as f:
                #f.write(model.get_printable_metrics() + "\n")

    #logger.info("Features set : " + set_name)
    #logger.info(Model.models_metric_summary(models))


def train_ml(model, X_train, y_train, X_test, y_test, set_name, folder, random=False, select_feature=True):
     
    model.train(X_train, y_train, random)
    model.predict(X_test, y_test)
    model.compute_metrics(folder, y_test)
    #logger.debug(model.get_printable_metrics())
    model.save(folder + '/'+ model.name + ".ckpt")   
    model.visualization()
    
    #with open(c.training_output_file, 'a') as f:
        #f.write("Features set : " + set_name + "\n")
        #f.write(model.get_printable_metrics() + "\n")
    #logger.info("Features set : " + set_name)
    #logger.info(model.get_printable_metrics())
