'''
Different functions that help with the evaluation of the models.
'''

import itertools
from graphviz import Digraph
from sklearn_crfsuite import metrics
from sklearn.metrics import log_loss, roc_curve, roc_auc_score

from IPython.display import HTML, Latex, display
import matplotlib.pyplot as plt


def print_classification_report(y_true, y_pred):
    '''[summary]

    Arguments:
        y_true {list} -- list of true labels e.g. [['O', 'I-Answer', 'O'], [...], ...]
        y_pred {list} -- list of predicted labels e.g. [['O', 'I-Answer', 'O'], [...], ...]
    '''
    print(metrics.flat_classification_report(y_true, y_pred, digits=3))

def visualize_transitions(crf_model):
    dot = Digraph(comment='The Transition Graph')
    for label in list(crf_model.classes_):
        dot.node(label, label)
    transitions = crf_model.transition_features_
    for transition in transitions.keys():
        dot.edge(transition[0], transition[1], label="%.2f" % transitions[transition])
    return dot

def print_annotated_text(token_list, y_preds, y_trues=None):
    if y_trues is None:
        y_trues = ["O"] * len(y_preds)
    output = ""
    for token, y_pred, y_true in zip(token_list, y_preds, y_trues):
        if y_true == "O":
            output += "<span title=\""+ \
                      str(y_pred["I-Answer"]+y_pred["B-Answer"])+ \
                      "\" style=\"background-color:rgba(255, 0, 0, "+ \
                      str(y_pred["I-Answer"]+y_pred["B-Answer"])+ ");\"> " + \
                      token + \
                      "</span>"
        else:
            output += "<span title=\""+ \
                      str(y_pred["I-Answer"]+y_pred["B-Answer"])+ \
                      "\" style=\"background-color:rgba(255, 0, 0, "+ \
                      str(y_pred["I-Answer"]+y_pred["B-Answer"])+ \
                      ");text-decoration: underline\"> " + \
                      token + \
                      "</span>"
    #print(output)
    display(HTML(output))

def print_annotated_text_latex(token_list, y_preds, y_trues=None):
    if y_trues is None:
        y_trues = ["O"] * len(y_preds)
    output = ""
    for token, y_pred, y_true in zip(token_list, y_preds, y_trues):
        if y_true == "O":
            output += "\colorbox{white!"+ str(100 - int((y_pred["I-Answer"]+y_pred["B-Answer"])*100)) + "!red}{" + token + "}"
        else:
            output += "\colorbox{white!"+ str(100 - int((y_pred["I-Answer"]+y_pred["B-Answer"])*100)) + "!red}{\\underline{" + token + "}}"
    #print(output)
    return output

def crf_roc_curve(y_trues, prob_pred):
    '''[summary]

    Arguments:
        y_true {nested list} -- list of true labels e.g. [['O', 'I-Answer', 'O'], [...], ...]
        prob_pred {[type]} -- [description]
    '''

    y_true_merged = [y_true == 'I-Answer' or y_true == 'B-Answer' for y_true in list(itertools.chain(*y_trues))]
    y_pred_list = [y_pred['I-Answer'] + y_pred['B-Answer'] for y_pred in list(itertools.chain(*prob_pred))]

    fpr, tpr, _ = roc_curve(y_true_merged, y_pred_list)
    auc = roc_auc_score(y_true_merged, y_pred_list)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='CRF')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve - AUC = {:0.2f}'.format(auc))
    plt.legend(loc='best')
    plt.show()
