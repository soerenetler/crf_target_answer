from collections import Counter
from itertools import chain

class numberBaseline:
    '''
    Baseline that identifies only numbers as questions-worthy
    '''
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        seq_list = []
        for x in X:
            seq = []
            for token in x:
                try:
                    print(token.keys())
                    if token["0:word.like_num"] == True:
                        seq.append("I")
                    else: 
                        seq.append("O")
                except KeyError:
                    seq.append("O")
            seq_list.append(seq)
        return seq_list

    def predict_proba(self, X):
        seq_list = []
        for x in X:
            seq = []
            for token in x:
                try:
                    if token["0:word.like_num"] == True:
                        seq.append({"I":1, "O":0})
                    else: 
                        seq.append({"I":0, "O":1})
                except KeyError:
                    seq.append({"I":0, "O":1})
            seq_list.append(seq)
        return seq_list

    def predict_marginals_single(self, x):
        seq = []
        for token in x:
            try:
                if token["0:word.like_num"] == True:
                    seq.append({"I":1, "O":0})
                else: 
                    seq.append({"I":0, "O":1})
            except KeyError:
                seq.append({"I":0, "O":1})
        return seq