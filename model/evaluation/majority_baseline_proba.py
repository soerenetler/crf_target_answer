from collections import Counter
from itertools import chain

class majorityProbaBaseline:
    '''
    Baseline that identifies every token as non-question-worthy with a confidence of 86 %
    '''
    def __init__(self):
        self.majority_class = "O"
        self.majority_class_proba = 0.86
        self.other_classes = ("I")

    def fit(self, _, y):
        y_merged = list(chain(*y))
        counter = Counter(y_merged)
        most_common, most_common_proba = counter.most_common(1)[0]
        self.majority_class = most_common
        self.majority_class_proba = most_common_proba/len(y_merged)
        keys = set(counter.keys())
        keys.discard(self.majority_class)
        self.other_classes = keys


    def predict(self, X):
        return [[self.majority_class] * len(X_seq) for X_seq in X]

    def predict_proba(self, X):
        prediction = {self.majority_class: self.majority_class_proba}
        for label in self.other_classes:
            prediction[label] = 1 -  self.majority_class_proba

        return [[prediction] * len(X_seq) for X_seq in X]

    def predict_marginals_single(self, x):
        prediction = {self.majority_class: self.majority_class_proba}
        for label in self.other_classes:
            prediction[label] = 1 - self.majority_class_proba
        return [prediction] * len(x)