
from pytorchnlp.utils.datahandler import DataHandlerDefault
import pytorchnlp.utils.prepro_utils
from pytorchnlp.utils import utils
from pytorchnlp.pytorch_utils import nn

import numpy as np
import pickle
import json


class DataHandlerSeqLabel(DataHandlerDefault):

	def __init__(self,task):
		self.lang = None #Lang('text', min_frequency=2)
		self.data_splits = {}
		self.tag_dct = None
		self.split_lens = None
		self.task = task
		self.loadPreproData()



	def loadPreproData(self, dump_dir="data/processed_data/"):
		# 	load vocab etc.
		prefix=""
		if self.task=="pos":
			prefix="pos_"
		self.lang = pickle.load(open(dump_dir+prefix+"lang.pickle","r"))
		self.tag_dct = pickle.load(open(dump_dir+prefix+"all_tags_indexer.pickle","r"))
		self.data_splits['train'] = json.load( open(dump_dir+prefix+"train.json", "r") )
		self.data_splits['val'] = json.load( open(dump_dir+prefix+"val.json", "r") )
		self.data_splits['test'] = json.load( open(dump_dir+prefix+"test.json", "r") )
		splits = self.data_splits.keys()
		self.split_lens = {split:len(self.data_splits[split]) for split in splits}
		self.tag_dct_reverse = {i:tag for tag,i in self.tag_dct.items()}
		print "split_lens = ", self.split_lens
		print "tag_dct = ", self.tag_dct


	def trialMode(self):
		# reduce train split, val split
		for split in self.data_splits:
			self.data_splits[split] = self.data_splits[split][:10]
		# update data lens
		splits = self.data_splits.keys()
		self.split_lens = {split:len(self.data_splits[split]) for split in splits}
		print "split_lens = ", self.split_lens


	### data info
	def getNumberOfBatches(self, split, batch_size):
		return ( self.split_lens[split] + batch_size - 1 ) / batch_size

	def getVocabSize(self):
		return len(self.lang.word2idx)

	def getTagSetSize(self):
		return len(self.tag_dct)


	### data utils
	def getTagsFromIndices(self, prediction_indices):
		'''
		prediction_indices: is a list of precitions indices
		ret: same dimensionalities. tags instead of indices
		'''
		return [self.tag_dct_reverse[tag] for tag in prediction_indices ]



	## batch and trianing
	def shuffleTrain(self):
		'''
		- shuffle th
		- Can also do bucketing by sentence length. And shuffle only within bucket
		'''
		indices = np.arange(self.split_lens['train'])
		np.random.shuffle(indices)
		self.data_splits['train']= [ self.data_splits['train'][idx] for idx in indices ]


	def getBatch(self, split, batch_size, i, get_all_vals=False, get_segments=False):
		'''
		Returns i'th batch from split, considering batch_size
		'''
		split_vals = self.data_splits[split][i*batch_size:(i+1)*batch_size]
		train_x =  [ self.lang.getSentenceToIndexList(' '.join(vals['text']) ) for vals in split_vals ]
		train_y =  [ [self.tag_dct[tag] for tag in vals['tags']] for vals in split_vals ]
		maxlen = max([len(train_x_i) for train_x_i in train_x])
		
		pad_token_lang = self.lang.getPadTokenIdx()
		pad_token = -1

		train_and_mask_x = [ utils.padSequence(train_x_i, maxlen, pad_token_lang, method="post") for train_x_i in train_x ]
		train_x = [ train_and_mask_x_i[0] for train_and_mask_x_i in train_and_mask_x ]
		#mask = [ train_and_mask_x_i[1] for train_and_mask_x_i in train_and_mask_x ]
		
		train_and_mask_y = [ utils.padSequence(train_y_i, maxlen, pad_token, method="post") for train_y_i in train_y ]
		## Note that -1 is pad token. This has ro be consistent with the loss function being used
		train_y = [ train_and_mask_y_i[0] for train_and_mask_y_i in train_and_mask_y ]
		mask = [ train_and_mask_x_i[1] for train_and_mask_x_i in train_and_mask_x ] # use mask of the predicted sequence class

		train_x = np.array( train_x )
		#print "train_x, train_y, mask = ", train_x, train_y, mask
		if get_all_vals:
			return train_x, train_y, mask, [vals['vals'] for vals in split_vals]
		return train_x, train_y, mask


	def _preproBatch(self):
		pass