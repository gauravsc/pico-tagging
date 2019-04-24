import os
import json
from pytorch_pretrained_bert import BertTokenizer
import numpy as np

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=512)

path = '../data/bioasq_dataset/train_data'
list_of_files = os.listdir(path)[0:500]
len_of_seqs = []
mesh_dict = {}
for file in list_of_files:
	data = json.load(open(path+'/'+file, 'r'))
	for article in data:
		mesh_assign = frozenset(article['mesh_labels'])
		if mesh_assign in mesh_dict:
			mesh_dict[mesh_assign] += 1
		else:
			mesh_dict[mesh_assign] = 1
		tokenized_text = tokenizer.tokenize('[CLS] '+article['abstract'].lower()+' [SEP]')
		len_of_seqs.append(len(tokenized_text))

len_of_seqs = np.array(len_of_seqs)
print("Percentage of abstracts with more than 512 tokens after (BERT) tokenization: ", np.sum(len_of_seqs>512)/len(len_of_seqs))

label_cnt = np.array(sorted(list(mesh_dict.values()), reverse=True))
label_cnt_frac = label_cnt/np.sum(label_cnt)
label_cnt_cumsum = np.cumsum(label_cnt_frac)

print ("Cumulative label count", label_cnt_cumsum)

