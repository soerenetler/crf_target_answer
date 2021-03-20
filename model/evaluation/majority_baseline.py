from collections import Counter
from itertools import chain

class majorityBaseline:
    '''
    Baseline that identifies every token as non-question-worthy
    '''
    def __init__(self):
        self.majority_class = "O"
        self.other_classes = ("I")

    def fit(self, X, y):
        y_merged = list(chain(*y))
        counter = Counter(y_merged)
        most_common, _ = counter.most_common(1)[0]
        self.majority_class = most_common
        self.other_classes = counter.keys().discard(self.majority_class)

    def predict(self, X):
        return [[self.majority_class] * len(X_seq) for X_seq in X]

    def predict_proba(self, X):
        prediction = {self.majority_class: 1}
        for label in self.other_classes:
            prediction[label] = 0

        return [[prediction] * len(X_seq) for X_seq in X]

    def predict_marginals_single(self, x):
        prediction = {self.majority_class: 1}
        for label in self.other_classes:
            prediction[label] = 0
        return [prediction] * len(x)