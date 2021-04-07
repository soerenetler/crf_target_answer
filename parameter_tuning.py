#!/usr/bin/env python
# coding: utf-8

import ast
import itertools
import pickle
from tqdm import tqdm
import spacy
from spacy.tokens import Doc
import pandas as pd
import joblib

from sklearn.metrics import make_scorer, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

import sklearn_crfsuite

from FeatureTransformer import FeatureTransformer

from sklearn.pipeline import Pipeline
from Custom import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameter Tuning for CRF model for target answer identification')
    parser.add_argument('--samples', dest='samples', metavar='s', type=int,default=None, 
                        help='number om samples')
    parser.add_argument('--data-folder',  dest='data', metavar='d', type=str, default="01_data/preprocessedData/", 
                        help='number om samples')
    parser.add_argument('--dataset',  dest='dataset', metavar='a', type=str, default="squad", 
                        help='number om samples')
    parser.add_argument('--output',  dest='output', metavar='o', type=str, default="./", 
                        help='number om samples')

    args = parser.parse_args()

    NO_SAMPLES = args.samples

    TRAIN_FILENAME = args.data +args.dataset+'/crf/'+str(NO_SAMPLES)+'_train_crf.txt'
    DEV_FILENAME = args.data +args.dataset+'/crf/'+str(NO_SAMPLES)+ '_test_crf.txt'

    import json
    with open(TRAIN_FILENAME, ) as file:
        train_data = json.load(file)

    with open(DEV_FILENAME, ) as file:
        dev_data = json.load(file)

    train_feature = train_data["x"] + dev_data["x"]
    
    y_train = train_data["y"] + dev_data["y"]
    y_train = io2iob(y_train)

    crf = Custom_CRF(algorithm='lbfgs',
                    max_iterations=100,
                    min_freq=5,
                    all_possible_transitions=True)

    tuned_parameters = {
        "c1":[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
        "c2":[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    }

    custom_roc_auc_scorer = make_scorer(custom_roc_auc_score, needs_proba=True)
    custom_log_loss_scorer = make_scorer(custom_log_loss, needs_proba=True)


    #Create CRF Pipeline
    gs = GridSearchCV(crf,
                      tuned_parameters,
                      n_jobs=-1,
                      cv=5,
                      scoring={"roc_auc": custom_roc_auc_scorer,
                               "log_loss": custom_log_loss_scorer},
                      refit="log_loss",
                      verbose=10)

    gs.fit(train_feature, y_train)

    joblib.dump(gs, args.output + "crf_target_answer/" + args.dataset + '/' + str(args.samples) + '/gs.pkl')
    joblib.dump(gs.best_estimator_, args.output + "crf_target_answer/" + args.dataset + '/' + str(args.samples) +'/best_estimator.pkl')