from sklearn.metrics import f1_score as scikit_f1_score
import numpy as np

def f1_score(true_label_mat, pred_label_mat):
	f1_score_micro = scikit_f1_score(true_label_mat, pred_label_mat, average='micro')
	f1_score_macro = scikit_f1_score(true_label_mat, pred_label_mat, average='macro')

	return f1_score_micro, f1_score_macro



