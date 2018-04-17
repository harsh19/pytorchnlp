# logging
import logging
logger = logging.getLogger(__name__)

# general python
import numpy as np
import scipy.misc
import pickle
import random
import math
import json

# pytorch
import torch

# lib
import utils


############################################################

class DataHandlerDefault:

	def __init__(self):
		# prepro objects
		pass

	###
	def preproData(self):
		# _preproData
		# _splitData
		pass


	def loadPreproData(self, data,):
		# 	load vocab etc.
		pass


	def trialMode(self):
		# reduce train split, val split
		# update data lens
		pass


	def _preproData(self, data):
		# 	dump vocab etc.
		pass


	def _splitData(self, data):
		# train,val,test splits
		# dump the spilits
		# data_lens, train_split
		pass


	### batch related
	def getNumberOfBatches(self, split, batch_size):
		return ( self.data_lens[split] + batch_size - 1 ) / batch_size


	def shuffleTrain(self):
		indices = np.arange(len(self.train_split))
		np.random.shuffle(indices)
		self.train_split= [ self.train_split[idx] for idx in indices ]


	def getBatch(self, split, batch_size, i):
		pass


	def _preproBatch(self):
		pass

