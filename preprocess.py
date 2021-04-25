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

def create_feature_json(input_file, output_file, no_samples, test=False):
        df_askable_paragraph = pd.read_csv(input_file)
        df_askable_paragraph["askable_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph["askable_tokens"]]
        df_askable_paragraph["sentence_tokens"] = [ast.literal_eval(t) for t in df_askable_paragraph["sentence_tokens"]]

        if not NO_SAMPLES is None:
            if not test:
                df = df_askable_paragraph.sample(n=NO_SAMPLES, random_state=1)
            else:
                if no_samples*0.2 > len(df_askable_paragraph):
                    df = df_askable_paragraph
                else:
                    df = df_askable_paragraph.sample(n=int(NO_SAMPLES*0.2), random_state=1)
        else:
            df = df_askable_paragraph


        feature = feature_transformer.transform(df["sentence_tokens"])
        y = list(df["askable_tokens"])
        tokens = list(df["sentence_tokens"])
        titles = list(df["text_title"])
        ids = list(df["paragraph_id"])

        with open(output_file, 'w') as outfile:
            json.dump({"x": feature, "y":y, "token": tokens, "title": titles, "id": ids} , outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameter Tuning for CRF model for target answer identification')
    parser.add_argument('--samples', dest='samples', metavar='s', type=int,default=None, 
                        help='number om samples')
    parser.add_argument('--data-folder',  dest='data', metavar='d', type=str, default="01_data/preprocessedData/", 
                        help='number om samples')
    parser.add_argument('--dataset',  dest='dataset', metavar='a', type=str, default="squad", 
                        help='number om samples')

    args = parser.parse_args()

    NO_SAMPLES=args.samples
    print(args.samples)

    TRAIN_FILENAME = args.data +args.dataset+ "/sentences/train.csv"
    DEV_FILENAME = args.data +args.dataset+ "/sentences/dev.csv"
    TEST_FILENAME = args.data +args.dataset+ "/sentences/test.csv"
    print(args.data)


    activated = spacy.require_gpu()
    nlp = spacy.load('en_core_web_trf')

    def custom_tokenizer(nlp):
        return lambda text : Doc(nlp.vocab, words=text)

    nlp.tokenizer = custom_tokenizer(nlp)

    feature_transformer = FeatureTransformer(nlp)

    create_feature_json(TRAIN_FILENAME, args.data +args.dataset+'/crf/'+str(NO_SAMPLES)+ '_train_crf.txt', NO_SAMPLES, test=False)
    create_feature_json(TEST_FILENAME, args.data +args.dataset+'/crf/'+str(NO_SAMPLES)+ '_test_crf.txt', NO_SAMPLES, test=True)