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

def prepare_minibatch(data, mesh_to_idx, tokenizer):
	X = []
	Y = []
	input_mask = []
	labels = []
	for article in data:
		# word_seq = article['abstract'].lower().strip().split(' ')
		tokenized_text = tokenizer.tokenize('[CLS] '+article['abstract'].lower())[0:512]
		idx_seq = tokenizer.convert_tokens_to_ids(tokenized_text)
		src_seq = np.zeros(max_seq_len)
		src_seq[0:len(idx_seq)] = idx_seq
		X.append(src_seq)
		mask = np.zeros(max_seq_len)
		mask[0:len(idx_seq)] = 1
		input_mask.append(mask)
		tgt_seq = np.zeros(len(mesh_to_idx))
		tgt_idx = [mesh_to_idx[mesh] for mesh in article['mesh_labels']]
		tgt_seq[tgt_idx] = 1
		Y.append(tgt_seq)
		labels.append(article['mesh_labels'])
	X = np.vstack(X)
	Y = np.vstack(Y)
	input_mask = np.vstack(input_mask)
	return X, input_mask, Y, labels


def train(model, criterion, mesh_to_idx, mesh_vocab, tokenizer):
	# read the list of files to be used for training
	path = '../data/bioasq_dataset/train_data'
	list_files = os.listdir(path)[0:200]
	print (list_files)
	best_f1_score = -100
	iters = 0
	list_losses = []
	for ep in range(max_epochs):
		model = model.train()
		for file in list_files:
			print("training file:", file)
			file_content = json.load(open(path+'/'+file, 'r'))
			i = 0
			while i < len(file_content):
				input_idx_seq, input_mask, target, labels = prepare_minibatch(file_content[i:i+batch_size], mesh_to_idx, tokenizer)
				
				# mask = np.zeros(target.shape)
				# mask[target==1] = 1
				# for k in range(target.shape[0]):
				# 	idx_zeros = np.random.choice(np.where(mask[k, :]==0)[0], 50*len(labels[k]))
				# 	# idx_zeros = np.where(mask[k, :]==0)[0]
				# 	mask[k, idx_zeros] = 1

				# mask = torch.tensor(mask).to(device, dtype=torch.float)

				input_idx_seq = torch.tensor(input_idx_seq).to(device, dtype=torch.long)
				input_mask = torch.tensor(input_mask).to(device, dtype=torch.long)
				target = torch.tensor(target).to(device, dtype=torch.float)
				output = model(input_idx_seq, input_mask)

				# computing the loss over the prediction
				loss = criterion(output, target)
				# loss = loss * mask
				# loss = torch.sum(loss, dim=(1))/torch.sum(mask, dim=(1))

				loss = torch.sum(loss, dim=(1))
				loss = torch.mean(loss)

				print ("loss: ", loss)

				# back-propagation
				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
				optimizer.step()

				i += batch_size
				# iters += 1

				list_losses.append(loss.data.cpu().numpy())

				# if iters % save_after_iters == 0:
				# 	model = model.eval()
				# 	f1_score_curr = validate(model, mesh_to_idx, mesh_vocab, tokenizer)
				# 	model = model.train()
				# 	print("Loss after ", iters, ":  ", np.mean(list_losses))
				# 	list_losses = []
				# 	if f1_score_curr > best_f1_score:
				# 		torch.save(model.state_dict(), '../saved_models/bert_based/model.pt')
				# 		best_f1_score = f1_score_curr

		for threshold in threshold_list:
			f1_score_curr = validate(model, mesh_to_idx, mesh_vocab, tokenizer, threshold)
			print ("F1 score: ", f1_score_curr, " at threshold: ", threshold)
			if f1_score_curr > best_f1_score:
				torch.save(model.state_dict(), '../saved_models/bert_based/full_model.pt')
				torch.save(model.bert.state_dict(), '../saved_models/bert_based/bert_retrained_mesh_model.pt')
				best_f1_score = f1_score_curr
		
		print("Loss after ", iters, ":  ", np.mean(list_losses))
		list_losses = []
		

	return model 


def validate(model, mesh_to_idx, mesh_vocab, tokenizer, threshold):
	model = model.eval()
	path = '../data/bioasq_dataset/val_data'
	list_files = os.listdir(path)
	print (list_files)

	true_labels = []
	pred_labels = []
	for file in list_files:
		file_content = json.load(open(path+'/'+file, 'r'))
		i = 0
		while i < len(file_content):
			input_idx_seq, input_mask, target, true_labels_batch = prepare_minibatch(file_content[i:i+4], mesh_to_idx, tokenizer)			
			input_idx_seq = torch.tensor(input_idx_seq).to(device, dtype=torch.long)
			input_mask = torch.tensor(input_mask).to(device, dtype=torch.long)
			predict = model(input_idx_seq, input_mask)
			predict = F.sigmoid(predict)
			predict[predict>threshold] = 1
			predict[predict<threshold] = 0
			predict = predict.data.to('cpu').numpy()

			for j in range(predict.shape[0]):
				nnz_idx = np.nonzero(predict[j, :])[0]
				pred_labels_article = [mesh_vocab[idx] for idx in nnz_idx]
				pred_labels.append(pred_labels_article)

			true_labels.extend(true_labels_batch)
			i += 4

	# for k in range(len(true_labels)):
	# 	print (true_labels[k])
	# 	print (pred_labels[k])

	f1_score_micro, f1_score_macro = f1_score(true_labels, pred_labels) 
	print ("f1 score micro: ", f1_score_micro, " f1 score macro: ", f1_score_macro)

	return f1_score_micro


if __name__ == '__main__':
	# location of the toy dataset ---> this needs to be replaced with the final dataset
	# data_file = '../data/bioasq_dataset/toyMeSH_2017.json'

	# # create the vocabulary for the input 
	# src_vocab, word_to_idx = get_vocab(data_file)
	# print("vocabulary of size: ", len(src_vocab))

	# Load pre-trained model tokenizer (vocabulary)
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=512)

	# create the vocabulary of mesh terms
	with open('../data/mesh_to_idx.pkl', 'rb') as fread:
		mesh_to_idx = pickle.load(fread)
	
	mesh_vocab = [" "] * len(mesh_to_idx)
	for mesh, idx in mesh_to_idx.items():
		mesh_vocab[idx] = mesh

	# setting different model parameters
	n_tgt_vocab = len(mesh_to_idx)
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

	train(model, criterion, mesh_to_idx, mesh_vocab, tokenizer)




