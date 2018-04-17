import pytorchnlp.utils.prepro as pytorchnlp_prepro
import pytorchnlp.utils.utils as pytorchnlp_utils



def getData(fname):
	data = open(fname, "r").readlines()
	m = len(data)
	i = 0
	ret = []
	while i<m:
		words = []
		tags = []
		all_vals = []
		while i<m:
			row = data[i]
			if len(row.strip())==0:
				break
			#print "row = ", "-"+row.strip()+ "-", len(row.strip())
			vals = row.strip().split('\t')
			if len(vals)<4:
				print "Error  for row = ", row
				i+=1
				continue
			word = vals[0]
			tag = vals[3]
			words.append(word)
			tags.append(tag)
			all_vals.append(vals)
			i+=1
		ret.append({'text':words, 'tags':tags, 'vals':all_vals} ) 
		i+=1
	return ret


def getNERData():
	train_data = getData("data/data/train.data")
	val_data = getData("data/data/dev.data")
	test_data = [] #getData("data/data/test_hid.data")
	return train_data, val_data, test_data