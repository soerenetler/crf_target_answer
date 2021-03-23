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

import sklearn_crfsuite

from FeatureTransformer import FeatureTransformer

from sklearn.pipeline import Pipeline
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

    NO_SAMPLES=args.samples
    print(args.samples)

    TRAIN_FILENAME = args.data +args.dataset+ "/sentences/train.csv"
    DEV_FILENAME = args.data +args.dataset+ "/sentences/dev.csv"
    TEST_FILENAME = args.data +args.dataset+ "/sentences/test.csv"
    print(args.data)

    df_askable_paragraph_train = pd.read_csv(TRAIN_FILENAME)
    df_askable_paragraph_train["askable_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph_train["askable_tokens"]]
    df_askable_paragraph_train["sentence_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph_train["sentence_tokens"]]

    df_askable_paragraph_test = pd.read_csv(TEST_FILENAME)
    df_askable_paragraph_test["askable_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph_test["askable_tokens"]]
    df_askable_paragraph_test["sentence_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph_test["sentence_tokens"]]

    nlp = spacy.load('en_core_web_trf')
    #spacy.require_gpu()
    
    def custom_tokenizer(nlp):
        return lambda text : Doc(nlp.vocab, words=text)

    nlp.tokenizer = custom_tokenizer(nlp)

    feature_transformer = FeatureTransformer(nlp)

    #Sample df
    if not NO_SAMPLES is None:
        df_train = df_askable_paragraph_train.sample(n=NO_SAMPLES, random_state=1)
        df_test = df_askable_paragraph_test.sample(n=int(NO_SAMPLES*0.2), random_state=1)
    else:
        df_train = df_askable_paragraph_train
        df_test = df_askable_paragraph_test


    train_feature = feature_transformer.fit_transform(df_train["sentence_tokens"])
    y_train = list(df_train["askable_tokens"])

    test_feature = feature_transformer.transform(df_test["sentence_tokens"])
    y_test = list(df_test["askable_tokens"])


    import json
    with open(args.data +args.dataset+'/crf/'+str(NO_SAMPLES)+'_train_crf.txt', 'w') as outfile:
        json.dump({"x": train_feature, "y":y_train} , outfile)

    with open(args.data +args.dataset+'/crf/'+str(NO_SAMPLES)+ '_test_crf.txt', 'w') as outfile:
        json.dump({"x": test_feature, "y":y_test} , outfile)