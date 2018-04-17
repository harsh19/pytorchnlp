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



####################################################################

class Lang():
	def __init__(self, name, min_frequency, consider_extra=True, text_prepro_func=None):
		self.name = name
		self.min_frequency = min_frequency
		self.consider_extra = consider_extra
		if self.consider_extra:
			self.pad_word = "<pad>"
			self.start = "<start>"
			self.end = "<end>"
			self.unk = "<unk>"
		self.word_frequencies = {}
		self.text_prepro_func = text_prepro_func

	def getStartTokenIdx(self):
		return self.word2idx[self.start]

	def getPadTokenIdx(self):
		return self.word2idx[self.pad_word]

	def getEndTokenIdx(self):
		return self.word2idx[self.end]

	def _sentenceToTokens(self, sentence):
		return utilities.splitIntoWords(sentence, do_prepro=True)

	def _addSentence(self, sentence):
		words=sentence.split() 
		#self._sentenceToTokens(sentence)
		for word in words:
			if word not in self.word_frequencies:
				self.word_frequencies[word]= 1
			else:
				self.word_frequencies[word]+= 1

	def getSentenceToIndexList(self, sentence):
		ret = []
		words = sentence.split()
		#words = self._sentenceToTokens(sentence)
		for word in words:
			if word in self.word2idx:
				ret.append(self.word2idx[word])
			else:
				ret.append(self.word2idx[self.unk])
		return ret

	def _getSentenceFromIndexList(self, lst_of_word_idx):
		ret = ' '.join( [self.idx2word[word_idx] for word_idx in lst_of_word_idx] )
		return ret.strip()

	def getSentenceFromIndexList(self, lst_of_word_idx, with_end_checks=True):
		if with_end_checks:
			ret = ''
			pad_token_id, end_token_id = self.getPadTokenIdx(), self.getEndTokenIdx()
			for word_idx in lst_of_word_idx:
				if word_idx == pad_token_id or word_idx == end_token_id:
					break
				ret = ret + self.idx2word[word_idx] + ' '
			return ret.strip()
		else:
			return self._getSentenceFromIndexList(lst_of_word_idx)
		

	def setupVocab(self, sentences):

		logger.info("="*33)
		logger.info("Lang: setupVocab()")

		# calculate frequencies
		map(self._addSentence,sentences)

		# init
		self.word2idx = {}
		if self.consider_extra:
			self.word2idx = {self.start:1,self.pad_word:0,self.end:2,self.unk:3}
		self.idx2word = {}
		self.vocab_cnt = len(self.word2idx)

		# setup vocab
		for word,freq in self.word_frequencies.items():
			if freq>=self.min_frequency:
				self.word2idx[word]= self.vocab_cnt
				self.vocab_cnt += 1
		self.idx2word = {idx:w for w,idx in self.word2idx.items()}

		logger.info("Vocab size = " + str(self.vocab_cnt) )


