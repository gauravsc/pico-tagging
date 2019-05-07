import json
import os
import pickle
import numpy as np
import random as rd
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer
from model.Models import CNNModel, BERTCLassifierModel, BERTClassifierLabelTransfer
from eval.eval import f1_score, precision_score, recall_score
from utils.embedding_operations import read_embeddings

# Global variables
batch_size = 4
clip_norm = 10.0
max_epochs = 50
device = 'cuda:0'
load_model = False
train_model = True
threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
aspect = 'population condition'

# set random seed
rd.seed(9001)
np.random.seed(9001)

def extract_target_vocab(data, aspect):
	vocab = []
	for sample in data:
		vocab += [triplet[2] for triplet in sample[aspect] if triplet[2] != "NULL"]
		# vocab += [triplet[2] for triplet in sample['intervention applied'] if triplet[2] != "NULL"]
		# vocab += [triplet[2] for triplet in sample['outcome condition'] if triplet[2] != "NULL"]
	idx_to_cui = list(set(vocab))

	cui_to_idx = {}
	for idx, cui in enumerate(idx_to_cui):
		cui_to_idx[cui] = idx

	return idx_to_cui, cui_to_idx 


def concept_cui_mapping(data, aspect):
	cui_to_concept = {}; concept_to_cui = {}
	for sample in data:
		all_triplets = sample[aspect]
		for triplet in all_triplets:
			if triplet[1] != 'NULL' and triplet[2] != 'NULL': 
				concept_to_cui[triplet[1]] = triplet[2]
				cui_to_concept[triplet[2]] = triplet[1]

	return cui_to_concept, concept_to_cui

def convert_to_doc_labels(Y, O, doc_idx, concepts, concept_to_cui, cui_to_idx):
	num_docs = len(list(set(doc_idx)))
	print ("num docs: ", num_docs, " num cuis: ", len(cui_to_idx), " max cui idx: ", np.max(list(cui_to_idx.values())), " max doc idx: ", np.max(doc_idx), " min doc idx: ", np.min(doc_idx))
	true_label_matrix = np.zeros((num_docs, len(cui_to_idx)))
	pred_label_matrix = np.zeros((num_docs, len(cui_to_idx)))

	for i, y in enumerate(Y):
		if y == 1:
			true_label_matrix[doc_idx[i], cui_to_idx[concept_to_cui[concepts[i]]]] = 1

	for i, y in enumerate(O):
		if y == 1:
			pred_label_matrix[doc_idx[i], cui_to_idx[concept_to_cui[concepts[i]]]] = 1

	return true_label_matrix, pred_label_matrix

# def display(results):
# 	print ("F1 Score Micro Population: ", f1_score_micro_p)
# 	print ("F1 Score Macro Population: ", f1_score_macro_p)
# 	print ("F1 Score Micro intervention: ", f1_score_micro_i)
# 	print ("F1 Score Macro intervention: ", f1_score_macro_i)
# 	print ("F1 Score Micro Outcome: ", f1_score_micro_o)
# 	print ("F1 Score Macro Outcome: ", f1_score_macro_o)

# 	print ("F1 Score Macro: ", (f1_score_macro_p+f1_score_macro_i+f1_score_macro_o)/3.0)
# 	print ("F1 Score Micro: ", (f1_score_micro_p+f1_score_micro_i+f1_score_micro_o)/3.0)

def prepare_data(data, cui_to_idx, tokenizer, for_test=False):
	X = []; Xt = []; Y = []; M = []; Mt = []; doc_idx = []; concepts = []; i = -1
	for article in data:
		input_text = article['population text'] + article['intervention text'] + article['outcome text']
		tokenized_text = tokenizer.tokenize('[CLS] ' + input_text.lower())[0:512]
		
		src_idx_seq = tokenizer.convert_tokens_to_ids(tokenized_text)
		src_seq = np.zeros(max_seq_len)
		src_seq[0:len(src_idx_seq)] = src_idx_seq

		# input padding mask 
		inp_mask = np.zeros(max_seq_len)
		inp_mask[0:len(src_idx_seq)] = 1

		positive_labels = [triplet[1] for triplet in article[aspect] if triplet[1] != "NULL"]
		
		if len(positive_labels) > 0:
			i += 1

		for label in positive_labels:
			tokenized_text = tokenizer.tokenize('[CLS] ' + label.lower())[0:10]
			tgt_idx_seq = tokenizer.convert_tokens_to_ids(tokenized_text)
			tgt_seq = np.zeros(10)
			tgt_seq[0:len(tgt_idx_seq)] = tgt_idx_seq
			Xt.append(tgt_seq)

			# input padding mask 
			tgt_mask = np.zeros(10)
			tgt_mask[0:len(tgt_idx_seq)] = 1
			Mt.append(tgt_mask)

			X.append(src_seq)
			M.append(inp_mask)
			Y.append(1)
			doc_idx.append(i)
			concepts.append(label)

		if for_test:
			negative_labels = [label for label in list(concept_to_cui.keys()) if label not in positive_labels]
			print ("size of concept to cui dict: ", len(concept_to_cui.keys()), "length of negative_labels: ", len(negative_labels))
		else:	
			negative_labels = rd.sample(list(concept_to_cui.keys()), 100*len(positive_labels))

		for label in negative_labels:
			tokenized_text = tokenizer.tokenize('[CLS] ' + label.lower())[0:10]
			tgt_idx_seq = tokenizer.convert_tokens_to_ids(tokenized_text)
			tgt_seq = np.zeros(10)
			tgt_seq[0:len(tgt_idx_seq)] = tgt_idx_seq
			Xt.append(tgt_seq)

			# input padding mask 
			tgt_mask = np.zeros(10)
			tgt_mask[0:len(tgt_idx_seq)] = 1
			Mt.append(tgt_mask)

			X.append(src_seq)
			M.append(inp_mask)
			Y.append(0)
			doc_idx.append(i)
			concepts.append(label)
	
	print ("Y-1: ", np.sum(Y), " Y-0", len(Y)-np.sum(Y))


	X = np.vstack(X)
	Xt = np.vstack(Xt)
	Y = np.vstack(Y)
	M = np.vstack(M)
	Mt = np.vstack(Mt)

	shuffled_indices = rd.sample(range(X.shape[0]), X.shape[0])
	X = X[shuffled_indices]
	Xt = Xt[shuffled_indices]
	Y = Y[shuffled_indices]
	M = M[shuffled_indices]
	Mt = Mt[shuffled_indices]
	# print ("before: ", doc_idx)
	doc_idx = [doc_idx[idx] for idx in shuffled_indices]
	# print ("after: ", doc_idx)
	concepts = [concepts[idx] for idx in shuffled_indices] 

	return X, Xt, Y, M, Mt, doc_idx, concepts


def train(model, train_data, val_data, criterion, cui_to_idx, idx_to_cui, tokenizer):	
	X, Xt, Y, M, Mt, _, _ = prepare_data(train_data, cui_to_idx, tokenizer)
	print ("Entered training function ...")
	print ("X shape: ", X.shape, " Xt Shape: ", Xt.shape, " Y Shape: ", Y.shape, " M Shape: ", M.shape, " Mt shape: ", Mt.shape)

	best_f1_score = -100
	list_losses = []	
	for ep in range(max_epochs):
		model = model.train()
		i = 0
		while i < X.shape[0]:
			x = torch.tensor(X[i:i+batch_size]).to(device, dtype=torch.long)
			m = torch.tensor(M[i:i+batch_size]).to(device, dtype=torch.long)
			xt = torch.tensor(Xt[i:i+batch_size]).to(device, dtype=torch.long)
			mt = torch.tensor(Mt[i:i+batch_size]).to(device, dtype=torch.long)
			y = torch.tensor(Y[i:i+batch_size]).to(device, dtype=torch.float)

			o = model(x, m, xt, mt)

			# computing the loss over the prediction
			loss = criterion(o, y)
			loss = torch.mean(loss)
			print ("loss: ", loss)

			# back-propagation
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
			optimizer.step()

			list_losses.append(loss.data.cpu().numpy())
			i += batch_size

		for threshold in threshold_list:
			f1_score_curr, _ = validate(model, val_data, cui_to_idx, tokenizer, threshold)
			print ("F1 score: ", f1_score_curr, " at threshold: ", threshold)
			if f1_score_curr > best_f1_score:
				torch.save(model.state_dict(), '../saved_models/bert_based/english_labels/'+aspect+'_'+'full_model.pt')
				# torch.save(model.bert.state_dict(), '../saved_models/bert_based/bert_retrained_mesh_model.pt')
				best_f1_score = f1_score_curr
		
		print("Loss after epochs ", ep, ":  ", np.mean(list_losses))
		list_losses = []
		
	return model 


def validate(model, data, cui_to_idx, tokenizer, threshold, test=False):
	model = model.eval()
	X, Xt, Y, M, Mt, doc_idx, concepts = prepare_data(data, cui_to_idx, tokenizer, for_test=test)
	# print ("Entered test function ...")
	# print ("X shape: ", X.shape, " Xt Shape: ", Xt.shape, " Y Shape: ", Y.shape, " M Shape: ", M.shape, " Mt shape: ", Mt.shape)

	O = []; i = 0
	while i < X.shape[0]:			
		x = torch.tensor(X[i:i+4]).to(device, dtype=torch.long)
		m = torch.tensor(M[i:i+4]).to(device, dtype=torch.long)
		xt = torch.tensor(Xt[i:i+4]).to(device, dtype=torch.long)
		mt = torch.tensor(Mt[i:i+4]).to(device, dtype=torch.long)
		
		o = model(x, m, xt, mt)
		o = F.sigmoid(o)

		o[o>=threshold] = 1
		o[o<threshold] = 0

		o = o.data.to('cpu').numpy().flatten()

		O.append(o)
		i += 4
		print (i,"/",X.shape[0])

	O = np.concatenate(O)


	Y, O = convert_to_doc_labels(Y, O, doc_idx, concepts, concept_to_cui, cui_to_idx)
	
	results = {}
	f1_score_micro, f1_score_macro = f1_score(Y, O)
	pr_score_micro, pr_score_macro = precision_score(Y, O)
	re_score_micro, re_score_macro = recall_score(Y, O)
	results['f1_score_micro'] = f1_score_micro
	results['f1_score_macro'] = f1_score_macro
	results['pr_score_micro'] = pr_score_micro
	results['pr_score_macro'] = pr_score_macro
	results['re_score_micro'] = re_score_micro
	results['re_score_macro'] = re_score_macro

	# display(results)

	return f1_score_micro, results


def tune_threshold(model, data, cui_to_idx, tokenizer):
	best_threshold = 0.0
	best_f1_score = -100
	for threshold in threshold_list:
		f1_score_curr, _ = validate(model, data, cui_to_idx, tokenizer, threshold)
		print ("F1 score: ", f1_score_curr, " at threshold: ", threshold)
		if f1_score_curr > best_f1_score:
				best_f1_score = f1_score_curr
				best_threshold = threshold
	return best_threshold


if __name__ == '__main__':
	# load the dataset
	data = json.load(open('../data/data_with_cuis.json', 'r'))
	# concept to cui mappings
	cui_to_concept, concept_to_cui = concept_cui_mapping(data, aspect)
	# # create the vocabulary for the input 
	idx_to_cui, cui_to_idx = extract_target_vocab(data, aspect)
	# Load pre-trained model tokenizer (vocabulary)
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=512)

	# load label count 
	label_cnt = json.load(open('../data/label_counts.json', 'r'))
	p_label_cnt = label_cnt['p_label_cnt']
	i_label_cnt = label_cnt['i_label_cnt']
	o_label_cnt = label_cnt['o_label_cnt']

	# Split train and test data
	train_idx = rd.sample(range(len(data)), int(0.8*len(data)))
	test_idx = [i for i in range(len(data)) if i not in train_idx]

	train_data = [data[i] for i in train_idx]
	test_data = [data[i] for i in test_idx]

	val_idx = rd.sample(range(len(train_data)), int(0.1*len(train_data)))
	train_idx = [i for i in range(len(train_data)) if i not in val_idx]
	val_data = [train_data[i] for i in val_idx]
	train_data = [train_data[i] for i in train_idx]

	# setting different model parameters
	n_tgt_vocab = len(cui_to_idx)
	max_seq_len = 512
	d_word_vec = 200
	dropout = 0.1
	learning_rate = 0.005

	model = BERTClassifierLabelTransfer(dropout=dropout)
	# model = nn.DataParallel(model, output_device=device)
	model.to(device)

	if load_model and os.path.isfile('../saved_models/bert_based/english_labels/'+aspect+'_'+'full_model.pt'):
		model.load_state_dict(torch.load('../saved_models/bert_based/english_labels/'+aspect+'_'+'full_model.pt'))
		print ("Done loading the saved model .....")

	criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
	# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.999))
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

	if train_model:
		model = train(model, train_data, val_data, criterion, cui_to_idx, idx_to_cui, tokenizer)

	# load the best performing model
	model.load_state_dict(torch.load('../saved_models/bert_based/english_labels/'+aspect+'_'+'full_model.pt'))
	best_threshold = tune_threshold(model, val_data, cui_to_idx, tokenizer)
	_, results = validate(model, test_data, cui_to_idx, tokenizer, best_threshold, test=True)


	print ("Results for ", aspect)
	print (results)



