# logging
import logging
logger = logging.getLogger(__name__)

# python
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import random
import math
import os

# pytorch
import torch.optim as optim
import torch.nn as nn
import torch

# other project files
from solver import Solver
from utils.prepro import Prepro

# argparse
import argparse

def parseArguments():

	parser=argparse.ArgumentParser()

	# mode
	parser.add_argument('-mode',dest='mode') # prepro, train, decode
	parser.add_argument('--debug', dest='debug', action='store_true', default=False)

	# data locations
	parser.add_argument('-data_dir',dest='data_dir', default='data/processed_data/') # transform, prepro, train

	# prepro and model params
	parser.add_argument('-min_frequency', dest='min_frequency', type=int, default=2)
	parser.add_argument('-emb_size', dest='emb_size', type=int, default=32)
	parser.add_argument('-hidden_size', dest='hidden_size', type=int, default=96)
	
	# train/test time params
	parser.add_argument('-optimizer_type', dest='optimizer_type', default='ADAM')
	parser.add_argument('-batch_size', dest='batch_size', type=int, default=4)
	parser.add_argument('-num_epochs', dest='num_epochs', type=int, default=10)
	parser.add_argument('-model_name', dest='model_name', default='default')
	parser.add_argument('-decoding_type', dest='decoding_type', default='greedy') # greedy, beam

	args=parser.parse_args()
	return args

params=parseArguments()
print params


def main():

	#import logging.config
	#logging.config.fileConfig('logs/logging.conf')

	if params.mode=="prepro":
		#---------------- preprocessing
		print "="*100
		prepro = Prepro()
		# create jsons. create splits. learn vocab. dump all relevant files
		prepro.preproData()
		# Sanity checks..

	elif params.mode=="train":
		#---------------- training
		print "="*100
		solver = Solver(params)
		solver.createModel()
		solver.train()

	elif params.mode=="decode":
		#---------------- decoding
		print "="*100
		solver = Solver(params)
		solver.loadSavedModel("./tmp/" + params.model_name + ".ckpt")
		if params.decoding_type=="beam":
			print "Overwriting batch size to be 1"
			params.batch_size = 1
		bleu = solver.getBleu(split="val", output_path="./tmp/" + params.model_name)
		print "val belu = ", bleu
		attn_weights_path = "./tmp/" + params.model_name + ".test" + ".attn_weights" + ".pickle"
		bleu = solver.getBleu(split="test", output_path="./tmp/" + params.model_name, save_attn_weights=True, attn_weights_path=attn_weights_path)
		print "test belu = ", bleu

	else:
		print "====== INVALID MODE ====="


main()
