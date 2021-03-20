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

if __name__ == "__main__":
    NO_SAMPLES=100

    TRAIN_FILENAME = "01_data/preprocessedData/random_train_sentences.csv"
    TEST_FILENAME = "01_data/preprocessedData/random_dev_sentences.csv"

    df_askable_paragraph_train = pd.read_csv(TRAIN_FILENAME)
    df_askable_paragraph_train["askable_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph_train["askable_tokens"]]
    df_askable_paragraph_train["sentence_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph_train["sentence_tokens"]]

    nlp = spacy.load('en_core_web_md')
    nlp.tokenizer = custom_tokenizer(df_askable_paragraph_train, nlp)

    feature_transformer = FeatureTransformer(nlp)

    crf = Custom_CRF(algorithm='lbfgs',
                    max_iterations=100,
                    min_freq=5,
                    all_possible_transitions=True)

    tuned_parameters = {
        "c1":[0.001, 0.01, 0.1, 1, 10, 100],
        "c2":[0.001, 0.01, 0.1, 1, 10, 100]
    }

    #Sample df
    
    if not NO_SAMPLES is None:
        df_train = df_askable_paragraph_train.sample(n=NO_SAMPLES, random_state=1)
    else:
        df_train = df_askable_paragraph_train
    
    train_feature = feature_transformer.fit_transform(df_train["sentence_text"])
    
    y_train = list(df_train["askable_tokens"])

    custom_roc_auc_scorer = make_scorer(custom_roc_auc_score, needs_proba=True)
    custom_log_loss_scorer = make_scorer(custom_log_loss, needs_proba=True)


    #Create CRF Pipeline
    gs = GridSearchCV(crf,
                      tuned_parameters,
                      cv=5,
                      scoring={"roc_auc": custom_roc_auc_scorer,
                               "log_loss": custom_log_loss_scorer},
                      refit="roc_auc",
                      verbose=10)

    gs.fit(train_feature, y_train)

    joblib.dump(gs, 'gs.pkl')
    joblib.dump(gs.best_estimator_, 'best_estimator.pkl')