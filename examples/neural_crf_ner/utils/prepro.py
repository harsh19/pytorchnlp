import pytorchnlp.utils.prepro_utils
from pytorchnlp.pytorch_utils.lang import Lang
import pickle
import json
import io


class Prepro():

	def __init__(self, params=None):
		self.lang = Lang('text', min_frequency=2)

	###
	def preproData(self):
		# _preproData
		# _splitData
		data = io.getNERData()
		self._splitData(data)
		self._preproData(self.data_splits['train'])
		self._dumpPrepro()


	def _preproData(self, data):
		# 	setup and dump vocab etc.
		sentences = [ ' '.join(row["text"]) for row in data] 
		self.lang.setupVocab(sentences)

	def _getTagsSet(self, dataset):
		ret = []
		for row in dataset:
			ret.extend(row['tags'])
		ret.append('start_tag')
		ret.append('stop_tag')
		return set(ret)

	def _splitData(self, data):
		# train,val,test splits
		# dump the spilits
		# data_lens, train_split
		train_data, val_data, test_data = data
		self.data_splits = {"train":train_data, "val":val_data, "test":test_data}
		splits = self.data_splits.keys()
		all_tags = self._getTagsSet(train_data)
		all_tags = [tag for tag in all_tags]
		self.all_tags_indexer = {tag:i for i,tag in enumerate(all_tags)}


	def _dumpPrepro(self, dump_dir="data/processed_data/"):
		json.dump( self.data_splits['train'], open(dump_dir+"train.json", "w") )
		json.dump( self.data_splits['val'], open(dump_dir+"val.json", "w") )
		json.dump( self.data_splits['test'], open(dump_dir+"test.json", "w") )
		pickle.dump( self.lang, open(dump_dir+"lang.pickle", "w") )
		pickle.dump( self.all_tags_indexer, open(dump_dir+"all_tags_indexer.pickle", "w") )





