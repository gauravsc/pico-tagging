import sys
import os
import json
import pickle

# Some global variables
file_size = 5000

def mesh_name_to_id_mapping():
	with open('../data/bioasq_dataset/MeSH_name_id_mapping_2017.txt','r') as f:
		mappings = f.readlines()
		mappings = [mapping.strip().split("=") for mapping in mappings]

	mappings_dict = {}
	for mapping in mappings:
		mappings_dict[mapping[0]] = mapping[1]

	return mappings_dict


# generatie batches and write them to a file
def generate_batches():

	with open('../../seq-to-tree/data/bioasq_dataset/allMeSH_2017.json', 'r', encoding="utf8", errors='ignore') as f:
		training_docs = json.load(f)['articles']
	
	english_name_to_mesh_id = mesh_name_to_id_mapping()	

	records = []
	for i, doc in enumerate(training_docs):
		abstract = doc['abstractText']
		mesh_labels = [english_name_to_mesh_id[mesh_name] for mesh_name in doc['meshMajor']]
		records.append({'abstract': abstract, 'mesh_labels': mesh_labels})

		# write the dara into file
		if (i+1) % file_size == 0:
			with open('../data/bioasq_dataset/train_data/'+str((i+1)//file_size)+'.json','w') as f:
				json.dump(records, f)
				records = []

	# write the left over data into file
	if len(records) > 0:
		with open('../data/bioasq_dataset/train_data/'+str(((i+1)//file_size) + 1) +'.json','w') as f:
			json.dump(records, f)


if __name__ == '__main__':
	generate_batches()
