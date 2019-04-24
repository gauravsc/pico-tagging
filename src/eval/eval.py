from sklearn.metrics import f1_score as scikit_f1_score
import numpy as np

def f1_score(true_labels, pred_labels):
	n_samples = len(true_labels)

	unique_labels = []
	for labels_one_abstract in true_labels:
		for label in labels_one_abstract:
			unique_labels.append(label)

	for labels_one_abstract in pred_labels:
		for label in labels_one_abstract:
			unique_labels.append(label)

	unique_labels = list(set(unique_labels))

	label_to_idx = {}
	for i in range(len(unique_labels)):
		label_to_idx[unique_labels[i]] = i

	true_label_mat = []
	pred_label_mat = []

	for i in range(n_samples):
		true_arr = np.zeros(len(unique_labels))
		pred_arr = np.zeros(len(unique_labels))

		idx_true = [label_to_idx[label] for label in true_labels[i]]
		idx_pred = [label_to_idx[label] for label in pred_labels[i]]

		true_arr[idx_true] = 1
		pred_arr[idx_pred] = 1

		true_label_mat.append(true_arr)
		pred_label_mat.append(pred_arr)


	true_label_mat = np.vstack(true_label_mat)
	pred_label_mat = np.vstack(pred_label_mat)

	f1_score_micro = scikit_f1_score(true_label_mat, pred_label_mat, average='micro')
	f1_score_macro = scikit_f1_score(true_label_mat, pred_label_mat, average='macro')


	# compute precision, recall and f1 scores for one instance

	return f1_score_micro, f1_score_macro



