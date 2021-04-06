import itertools
from sklearn.metrics import make_scorer, log_loss, roc_auc_score
from spacy.tokens import Doc
from sklearn_crfsuite import CRF

def custom_tokenizer(nlp):
    return lambda text : Doc(nlp.vocab, words=text)
        
def custom_roc_auc_score(y_trues, prob_pred):
    y_true_merged = [y_true == 'I-Answer' for y_true in list(itertools.chain(*y_trues))]
    y_pred_list = [y_pred['I-Answer'] for y_pred in list(itertools.chain(*prob_pred))]

    return roc_auc_score(y_true_merged, y_pred_list)

def custom_log_loss(y_trues, prob_pred):
    y_true_merged = [y_true == 'I-Answer' for y_true in list(itertools.chain(*y_trues))]
    y_pred_list = [y_pred['I-Answer'] for y_pred in list(itertools.chain(*prob_pred))]
    
    return log_loss(y_true_merged, y_pred_list)

def io2iob(y):
    y_iob = []
    for sequence in y:
        previous_label = "O"
        sequence_iob = []
        for label in sequence:
            if privious_label === "O" and label == "I-Answer"
                sequence_iob.append("B-Answer")
            else:
                sequence_iob.append(label)

        y_iob.appen(sequence_iob)

    return y_iob


class Custom_CRF(CRF):
    def predict_proba(self, X):
        return self.predict_marginals(X)