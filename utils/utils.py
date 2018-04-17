import random


### masking and padding
def maskSequence(seq, desired_length, pad_symbol, method="pre"):
	seq_length=len(seq)
	mask=[1,]*desired_length
	if len(seq)<desired_length:
		if method=="post":
			mask=[1,]*seq_length+[0,]*(desired_length-seq_length)
		else:
			mask=[0,]*(desired_length-seq_length)+[1,]*seq_length
	return mask

def padSequence(seq, desired_length, pad_symbol, method="pre"):
	seq_length=len(seq)
	mask=[1,]*desired_length
	if len(seq)<desired_length:
		if method=="post":
			seq=seq+[pad_symbol,]*(desired_length-seq_length)
			mask=[1,]*seq_length+[0,]*(desired_length-seq_length)
		else:
			seq=[pad_symbol,]*(desired_length-seq_length)+seq
			mask=[0,]*(desired_length-seq_length)+[1,]*seq_length
	return seq, mask


### distribution utils
def sampleFromDistribution(vals, unnormalized=False):
	if unnormalized:
		vals =  np.exp(vals) / np.sum( np.exp(vals) )
	p = random.random()
	s=0.0
	for i,v in enumerate(vals):
		s+=v
		if s>=p:
			return i
	return len(vals)-1
