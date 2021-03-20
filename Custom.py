import itertools
from sklearn.metrics import make_scorer, log_loss, roc_auc_score
from spacy.tokens import Doc
import spacy
from sklearn_crfsuite import CRF

def custom_tokenizer(df, nlp):
    return lambda text : Doc(nlp.vocab, list(df[df["paragraph_text"]==text]["paragraph_tokens"])[0])
        
def custom_roc_auc_score(y_trues, prob_pred):
    y_true_merged = [y_true == 'I-Answer' for y_true in list(itertools.chain(*y_trues))]
    y_pred_list = [y_pred['I-Answer'] for y_pred in list(itertools.chain(*prob_pred))]

    return roc_auc_score(y_true_merged, y_pred_list)

def custom_log_loss(y_trues, prob_pred):
    y_true_merged = [y_true == 'I-Answer' for y_true in list(itertools.chain(*y_trues))]
    y_pred_list = [y_pred['I-Answer'] for y_pred in list(itertools.chain(*prob_pred))]
    
    return log_loss(y_true_merged, y_pred_list)

class Custom_CRF(CRF):
    def predict_proba(self, X):
        return self.predict_marginals(X)