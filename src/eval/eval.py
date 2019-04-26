from sklearn.metrics import f1_score as scikit_f1_score
from sklearn.metrics import precision_score as scikit_precision_score
from sklearn.metrics import recall_score as scikit_recall_score
import numpy as np

def f1_score(true_label_mat, pred_label_mat):
	f1_score_micro = scikit_f1_score(true_label_mat, pred_label_mat, average='micro')
	f1_score_macro = scikit_f1_score(true_label_mat, pred_label_mat, average='macro')

	return f1_score_micro, f1_score_macro


def precision_score(true_label_mat, pred_label_mat) :
	precision_score_micro = scikit_precision_score(true_label_mat, pred_label_mat, average='micro')
	precision_score_macro = scikit_precision_score(true_label_mat, pred_label_mat, average='macro')

	return precision_score_micro, precision_score_macro


def recall_score(true_label_mat, pred_label_mat):
	recall_score_micro = scikit_recall_score(true_label_mat, pred_label_mat, average='micro')
	recall_score_macro = scikit_recall_score(true_label_mat, pred_label_mat, average='macro')

	return recall_score_micro, recall_score_macro