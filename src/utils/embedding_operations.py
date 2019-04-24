import numpy as np

def read_embeddings(file, vocab):
	word_to_ind = {}
	for i, word in enumerate(vocab):
		word_to_ind[word] = i

	with open(file, 'r') as f:
		data = f.readlines()
	
	lines = data[1:]

	word_emb_dict = {}
	for line in lines:
		line = line.strip().split(' ')
		word = line[0]
		embedding = [float(v) for v in line[1:]]
		if word in word_to_ind:
			word_emb_dict[word] = embedding

	emb_mat = []
	for i in range(len(vocab)):
		if vocab[i] in word_emb_dict:
			if i == 0:
				emb_mat.append(np.zeros_like(word_emb_dict[vocab[i]]))
			else:
				emb_mat.append(np.array(word_emb_dict[vocab[i]]))
		else:
			emb_mat.append(np.array(np.random.standard_normal(len(embedding)).tolist()))

	emb_mat = np.vstack(emb_mat)

	return emb_mat

