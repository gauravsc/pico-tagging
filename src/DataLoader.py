import json


class DataLoader():
	def __init__(self, path, batch_size=32):
		self.files = os.listdir(path)
		self.file_ctr = 0
		self.batch_ctr = 0

	def next_batch():

		

