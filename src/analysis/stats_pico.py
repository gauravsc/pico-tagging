import json

# extract vocabulary from the data file
def extract_target_vocab(data):
	vocab = []
	for sample in data:
		vocab += [triplet[2] for triplet in sample['population condition'] if triplet[2] is not "NULL"]
		vocab += [triplet[2] for triplet in sample['intervention applied'] if triplet[2] is not "NULL"]
		vocab += [triplet[2] for triplet in sample['outcome condition'] if triplet[2] is not "NULL"]
	idx_to_cui = list(set(vocab))

	cui_to_idx = {}
	for idx, cui in enumerate(idx_to_cui):
		cui_to_idx[cui] = idx

	return idx_to_cui, cui_to_idx 


def update_label_count(label_cnt, labels):
	for label in labels:
		if label in label_cnt:
			label_cnt[label] += 1
		else:
			label_cnt[label] = 1
	return label_cnt


data = json.load(open('../data/data_with_cuis.json', 'r')) 
# # create vocabulary for the input 
idx_to_cui, cui_to_idx = extract_target_vocab(data)

p_label_cnt = {}; i_label_cnt = {}; o_label_cnt = {}
for article in data:
	p_labels = [triplet[2] for triplet in article['population condition'] if triplet[2] != "NULL"]
	p_label_cnt = update_label_count(p_label_cnt, p_labels)
	i_labels = [triplet[2] for triplet in article['intervention applied'] if triplet[2] != "NULL"]
	i_label_cnt = update_label_count(i_label_cnt, i_labels)
	o_labels = [triplet[2] for triplet in article['outcome condition'] if triplet[2] != "NULL"]
	o_label_cnt = update_label_count(o_label_cnt, o_labels)

json.dump({'p_label_cnt':p_label_cnt, 'i_label_cnt': i_label_cnt, 'o_label_cnt': o_label_cnt}, open('../data/label_counts.json', 'w'))

print ("unique labes in P: ", len(p_label_cnt), " I: ", len(i_label_cnt), " O: ", len(o_label_cnt))

print ("all labels in P: ", sum(p_label_cnt.values()), " I: ", sum(i_label_cnt.values()), " O: ", sum(o_label_cnt.values()))

print ("Total Instances: ", len(data)) 