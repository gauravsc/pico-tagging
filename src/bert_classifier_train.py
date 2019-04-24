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
from model.Models import CNNModel, BERTCLassifierModel
from eval.eval import f1_score
from utils.embedding_operations import read_embeddings

# Global variables
batch_size = 4
clip_norm = 10.0
max_epochs = 100
device = 'cuda:0'
load_model = False
threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# set random seed
rd.seed(9001)
np.random.seed(9001)

def extract_target_vocab(data):
	vocab = []
	for sample in data::
		vocab += [triplet[2] for triplet in sample['population condition'] if triplet[2] is not "NULL"]
		vocab += [triplet[2] for triplet in sample['intervention applied'] if triplet[2] is not "NULL"]
		vocab += [triplet[2] for triplet in sample['outcome condition'] if triplet[2] is not "NULL"]
	idx_to_cui = list(set(vocab))

	cui_to_idx = {}
	for idx, cui in enumerate(idx_to_cui):
		cui_to_idx[cui] = idx

	return idx_to_cui, cui_to_idx 

def display(f1_score_micro_p, f1_score_macro_p, f1_score_micro_i, f1_score_macro_i, f1_score_micro_o, f1_score_macro_o):
	print ("F1 Score Micro Population: ", f1_score_micro_p)
	print ("F1 Score Macro Population: ", f1_score_macro_p)
	print ("F1 Score Micro intervention: ", f1_score_micro_i)
	print ("F1 Score Macro intervention: ", f1_score_macro_i)
	print ("F1 Score Micro Outcome: ", f1_score_micro_o)
	print ("F1 Score Macro Outcome: ", f1_score_macro_o)

	print ("F1 Score Macro: ", (f1_score_macro_p+f1_score_macro_i+f1_score_macro_o)/3.0)
	print ("F1 Score Micro: ", (f1_score_micro_p+f1_score_micro_i+f1_score_micro_o)/3.0)

def prepare_data(data, cui_to_idx, tokenizer):
	X = []
	Y_p = []
	Y_i = []
	Y_o = []
	Mask = []
	
	for article in data:
		input_text = article['population text'] + article['intervention text'] + article['outcome text']
		tokenized_text = tokenizer.tokenize('[CLS] '+article['abstract'].lower())[0:512]
		idx_seq = tokenizer.convert_tokens_to_ids(tokenized_text)
		src_seq = np.zeros(max_seq_len)
		src_seq[0:len(idx_seq)] = idx_seq
		X.append(src_seq)
		
		# input padding mask 
		mask = np.zeros(max_seq_len)
		mask[0:len(idx_seq)] = 1
		Mask.append(mask)

		# population target
		tgt_seq_p = np.zeros(len(cui_to_idx))
		tgt_idx_p = [cui_to_idx[triplet[2]] for triplet in article['population condition'] if triplet[2] is not "NULL"]
		tgt_seq_p[tgt_idx_p] = 1
		Y_p.append(tgt_seq_p)

		# intervention target
		tgt_seq_i = np.zeros(len(cui_to_idx))
		tgt_idx_i = [cui_to_idx[triplet[2]] for triplet in article['intervention applied'] if triplet[2] is not "NULL"]
		tgt_seq_i[tgt_idx_i] = 1
		Y_i.append(tgt_seq_i)

		# outcome target
		tgt_seq_o = np.zeros(len(cui_to_idx))
		tgt_idx_o = [cui_to_idx[triplet[2]] for triplet in article['outcome condition'] if triplet[2] is not "NULL"]
		tgt_seq_o[tgt_idx_o] = 1
		Y_o.append(tgt_seq_o)

	X = np.vstack(X)
	Y_p = np.vstack(Y_p)
	Y_i = np.vstack(Y_i)
	Y_o = np.vstack(Y_o)
	Mask = np.vstack(Mask)
	
	return X, Mask, Y_p, Y_i, Y_o

def train(model, data, criterion, cui_to_idx, idx_to_cui, tokenizer):
	X, Mask, Y_p, Y_i, Y_o = prepare_data(data)
	
	best_f1_score = -100
	list_losses = []	
	for ep in range(max_epochs):
		model = model.train()
			while i < len(X.shape[0]):
				input_idx_seq = torch.tensor(X[i:i+batch_size]).to(device, dtype=torch.long)
				input_mask = torch.tensor(Mask[i:i+batch_size]).to(device, dtype=torch.long)
				target_p = torch.tensor(Y_p[i:i+batch_size]).to(device, dtype=torch.float)
				target_i = torch.tensor(Y_i[i:i+batch_size]).to(device, dtype=torch.float)
				target_o = torch.tensor(Y_o[i:i+batch_size]).to(device, dtype=torch.float)
				output_p, output_i, output_o = model(input_idx_seq, input_mask)

				# computing the loss over the prediction
				loss = (criterion(output_p, target_p) + criterion(output_i, target_i) + criterion(output_o, target_o))*1/3.0
				loss = torch.sum(loss, dim=(1))
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
			f1_score_curr = validate(model, mesh_to_idx, mesh_vocab, tokenizer, threshold)
			print ("F1 score: ", f1_score_curr, " at threshold: ", threshold)
			if f1_score_curr > best_f1_score:
				torch.save(model.state_dict(), '../saved_models/bert_based/full_model.pt')
				# torch.save(model.bert.state_dict(), '../saved_models/bert_based/bert_retrained_mesh_model.pt')
				best_f1_score = f1_score_curr
		
		print("Loss after ", iters, ":  ", np.mean(list_losses))
		list_losses = []
		
	return model 


def validate(model, mesh_to_idx, mesh_vocab, tokenizer, threshold):
	model = model.eval()
	X, Mask, Y_p, Y_i, Y_o = prepare_data(data)

	true_labels_mat_p = []
	pred_labels_mat_p = []
	true_labels_mat_i = []
	pred_labels_mat_i = []
	true_labels_mat_o = []
	pred_labels_mat_o = []
	
	i = 0
	while i < X.shape[0]:			
		input_idx_seq = torch.tensor(X[i:i+4]).to(device, dtype=torch.long)
		input_mask = torch.tensor(Mask[i:i+4]).to(device, dtype=torch.long)
		predict_p, predict_i, predict_o = model(input_idx_seq, input_mask)
		
		predict_p = F.sigmoid(predict_p)
		predict_i = F.sigmoid(predict_i)
		predict_o = F.sigmoid(predict_o)
		
		predict_p[predict_p>threshold] = 1
		predict_p[predict_p<threshold] = 0
		predict_i[predict_i>threshold] = 1
		predict_i[predict_i<threshold] = 0
		predict_o[predict_o>threshold] = 1
		predict_o[predict_o<threshold] = 0
		
		predict_p = predict_p.data.to('cpu').numpy()
		predict_i = predict_i.data.to('cpu').numpy()
		predict_o = predict_o.data.to('cpu').numpy()
		
		true_labels_mat_p.append(target_p)
		true_labels_mat_i.append(target_i)
		true_labels_mat_o.append(target_o)

		pred_labels_mat_p.append(predict_p)
		pred_labels_mat_i.append(predict_i)
		pred_labels_mat_o.append(predict_o)

		i += 4

	true_labels_mat_p = np.vstack(true_labels_mat_p)
	true_labels_mat_i = np.vstack(true_labels_mat_i)
	true_labels_mat_o = np.vstack(true_labels_mat_o)

	pred_labels_mat_p = np.vstack(pred_labels_mat_p)
	pred_labels_mat_i = np.vstack(pred_labels_mat_i)
	pred_labels_mat_o = np.vstack(pred_labels_mat_o)

	f1_score_micro_p, f1_score_macro_p = f1_score(true_labels_mat_p, pred_labels_mat_p)
	f1_score_micro_i, f1_score_macro_i = f1_score(true_labels_mat_i, pred_labels_mat_i) 
	f1_score_micro_o, f1_score_macro_o = f1_score(true_labels_mat_o, pred_labels_mat_o) 

	display(f1_score_micro_p, f1_score_macro_p, f1_score_micro_i, f1_score_macro_i, f1_score_micro_o, f1_score_macro_o)

	return (f1_score_micro_p + f1_score_micro_i + f1_score_micro_o)/3.0


if __name__ == '__main__':
	# load the dataset
	data = pickle.load(open('../data/data_with_cuis.pkl', 'r')) 
	# # create the vocabulary for the input 
	idx_to_cui, cui_to_idx = extract_target_vocab(data)
	# Load pre-trained model tokenizer (vocabulary)
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=512)

	# Split train and test data
	train_idx = rd.sample(range(len(data)), int(0.7*len(data)))
	test_idx = [i for i in range(len(data)) if i not in train_idx]

	train_data = [data[i] for i in range(train_idx)]
	test_idx = [data[i] for i in range(test_idx)]

	# setting different model parameters
	n_tgt_vocab = len(cui_to_idx)
	max_seq_len = 512
	d_word_vec = 200
	dropout = 0.1
	learning_rate = 0.005

	model = BERTCLassifierModel(n_tgt_vocab, dropout=dropout)
	# model = nn.DataParallel(model, output_device=device)
	model.to(device)

	if load_model and os.path.isfile('../saved_models/bert_based/model.pt'):
		model.load_state_dict(torch.load('../saved_models/bert_based/model.pt'))
		print ("Done loading the saved model .....")

	criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
	# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.999))
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

	train(model, criterion, cui_to_idx, idx_to_cui, tokenizer)




