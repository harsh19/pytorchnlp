# logging
import logging
logger = logging.getLogger(__name__)

# python
import numpy as np
import scipy.misc
import random
import math
import os

# pytorch
import torch.optim as optim
import torch.nn as nn
import torch

# image and plots
import numpy as np

# other project files
import utils.io
import utils.utilities
from models import rnn_model
from models import birnn_crf_model
import scores

# lib
from utils.datahandler import DataHandlerSeqLabel as DataHandler


class Solver:

	'''
	prepro: preprocessAll
	train: loadPrepro, createModel, train
	test: loadPrepro, createModel, loadSavedModel, decode
	'''

	def __init__(self, params):

		self.params = params

		#---- create data handler
		print "Creating Data Handler object"
		self.data_handler = DataHandler()
		self.params.vocab_size = self.data_handler.getVocabSize()
		self.params.tag_size = self.data_handler.getTagSetSize()
		if params.debug:
			self.data_handler.trialMode()


	def preprocessAll(self, data):
		self.data_handler.preproData(data)


	def loadPrepro(self):
		self.data_handler.loadPrepro()


	def createModel(self):

		#---- create model
		print ":---- create model", self.params.model_type
		params = self.params
		if params.model_type=="rnn" or params.model_type=="birnn":
			self.model = rnn_model.RNNTagger(self.params, params.model_type)
		else:
			self.model = birnn_crf_model.BiRNNCRF(self.params, self.data_handler.tag_dct, params.model_type)
		
		# loss function
		self.loss_function = nn.NLLLoss(ignore_index=-1,size_average=False )
		#self.loss_function = nn.NLLLoss(size_average=False ) # if ignore_index=0, then it wastes parametrrs, and kind of makes inaccurate softmnax decisions. use masking instead
		if torch.cuda.is_available():
			logger.info("CUDA AVAILABLE. Making adjustments")
			self.model.cuda()
			self.loss_function = self.loss_function.cuda()
		
		# optimizer
		logger.info( "DEFINING OPTIMIZER" )
		self.optimizer=None
		if params.optimizer_type=="SGD":
			self.optimizer = optim.SGD(self.model.parameters(),lr=0.05)
		elif params.optimizer_type=="ADAM":
			self.optimizer = optim.Adam(self.model.parameters())
		else:
			print "unsupported choice of optimizer"


	def train(self):

		logger.info("="*33)
		logger.info("Beginning training procedure")

		train_len = self.data_handler.split_lens['train']
		train_batch_for_qual = self.data_handler.getBatch(split='train', batch_size=self.params.batch_size,i=0 )

		for epoch in range(self.params.num_epochs):

			print "\n ------------- Epoch = "+str(epoch) + "-"*21 + "\n"

			epoch_loss = 0.0
			mask_y_sum = 0.0
			num_batches = self.data_handler.getNumberOfBatches('train', self.params.batch_size)
			for batch_idx in range(num_batches):
				batch_x, batch_y, mask = self.data_handler.getBatch(split='train',batch_size=self.params.batch_size,i=batch_idx)
				deb=False
				if batch_idx==0:
					deb=True
				loss = self._trainBatch(batch_x, batch_y, deb=deb)
				epoch_loss+=loss
				mask_y_sum += np.sum(mask)
				if batch_idx%self.params.print_batch_freq==0:
					print "Mean train loss after ",batch_idx,"batches of",epoch," epochs ="+str(epoch_loss/mask_y_sum)

			if self.params.model_type in ["rnn","birnn"]:
				num_batches = self.data_handler.getNumberOfBatches('val', self.params.batch_size)
				val_loss = 0.0
				mask_y_sum = 0.0
				for i in range(num_batches):
					batch_x, batch_y, mask_y = self.data_handler.getBatch(split='val', batch_size=self.params.batch_size, i=i)
					val_loss+= self._getLoss(batch_x, batch_y)
					mask_y_sum += np.sum(mask_y)
				print "Epoch val loss = "+str(val_loss)
				print "Epoch val perplexity = "+str( np.exp(val_loss/mask_y_sum) )
			
			# check validation accuracy
			self._evaluateAccuracy('val', self.params.model_name+"_"+str(epoch))			

			# save model
			if epoch%self.params.save_epoch_freq==0:
				self._saveModel(str(epoch))
			
			# shuffle training
			self.data_handler.shuffleTrain()


	def loadSavedModel(self, model_path):
		checkpoint = torch.load(model_path)
		self.model.load_state_dict({k:v for k,v in checkpoint.items() if k!="optimizer"})
		#if optimizer!=None:
		#	optimizer.load_state_dict(checkpoint['optimizer'])
		print "Loaded Model"


	###
	def _saveModel(self, extra):
		checkpoint = self.model.state_dict()
		#checkpoint['optimizer'] = self.optimizer.state_dict()
		torch.save(checkpoint, "./tmp/"+self.params.model_name+"_"+extra+".ckpt")
		print "Saved Model"

	def _trainBatch(self, batch_x, batch_y, deb=False):
		self.model.zero_grad()
		self.optimizer.zero_grad()
		#loss = self.model(batch_x, gt_output=batch_y, mode='train', loss_function=self.loss_function)
		loss = self.model(inp=batch_x, gt_output=batch_y, mode='train', loss_function=self.loss_function, debug=deb)
		#print "loss = ",loss
		loss.backward()
		self.optimizer.step()
		return loss.data[0]

	def _getLoss(self, batch_x, batch_y):
		loss = self.model(inp=batch_x, gt_output=batch_y, mode='train', loss_function=self.loss_function)
		return loss.data[0]
	
	def _decodeBatch(self, batch_x, batch_y=None, mask=None):
		#print "batch_y = ",batch_y
		_,outputs = self.model( inp=batch_x, gt_output=batch_y, mask=mask, mode='decode', loss_function=self.loss_function, get_loss=True)
		return outputs

	def _decodeAll(self, split="val"): # 'val'
		all_outputs = []
		all_gt = []
		all_vals = []
		num_batches = self.data_handler.getNumberOfBatches(split, self.params.batch_size)
		for batch_idx in range(num_batches):
			batch_x, batch_y, mask, batch_vals = self.data_handler.getBatch(split=split, batch_size=self.params.batch_size, i=batch_idx, get_all_vals=True)
			outputs = self._decodeBatch(batch_x, batch_y, mask)
			all_outputs.extend(outputs)
			all_gt.extend(batch_y)
			all_vals.extend(batch_vals)
		return all_outputs, all_gt, all_vals

	def _evaluateAccuracy(self, split='val', fname="val"):
		all_outputs, all_gt, all_vals = self._decodeAll(split=split)
		all_outputs = map(self.data_handler.getTagsFromIndices, all_outputs)
		#print "all_vals = ", all_vals
		#print "all_outputs = ", all_outputs
		out_data = []
		for output,vals in zip(all_outputs,all_vals):
			for val,out in zip(vals,output):
				out_data.append(' '.join(val)+' '+out)
			out_data.append(' ')
		fname = 'tmp/'+fname+".predictions"
		self._outputToFile(fname, out_data)
		print "SCORES = ", scores.scores(fname)

	def _outputToFile(self, fpath, out_data):
		fw = open(fpath,"w")
		for row in out_data:
			row = row.strip()
			fw.write(row + "\n")
		fw.close()

