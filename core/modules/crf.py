# pytorch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

# python general
import numpy as np
import random
import math

#mylibrary
from pytorchnlp.pytorch_utils import nn as pyutils

negative_inf = -99999999.0


class NeuralCRF(nn.Module):


    def __init__(self, vocab_size, tag_to_idx_mapper, start_tag, stop_tag ):
        super(NeuralCRF, self).__init__()
        self.vocab_size = vocab_size # vocab_size: this is tagset vocab size
        self.tag_to_idx_mapper = tag_to_idx_mapper
        self.tagset_size = len(tag_to_idx_mapper)
        self.start_tag = start_tag
        self.stop_tag = stop_tag
        self.transition_potentials = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))#transitions[]
        self.transition_potentials.data[tag_to_idx_mapper[start_tag], :] = negative_inf
        self.transition_potentials.data[:, tag_to_idx_mapper[stop_tag]] = negative_inf
        print "NeuralCRF modules: ",self._modules.keys()


    # for debugging and observaing changes in transition
    def printTransitionParams(self): 
        print "self.transition_potentials.data.cpu().numpy(): "
        print "\t",
        tagset = self.tag_to_idx_mapper.keys()
        for tag in tagset:
            print tag,"\t",
        print ""
        for i,row in  enumerate(self.transition_potentials.data.cpu().numpy()):
            print tagset[i],"\t",
            for col in row:
                print "%.2f\t" % col,
            print ""


    def computePartition(self, unary_potentials):
        last_step_values = torch.Tensor(1, self.tagset_size).fill_(negative_inf)
        last_step_values[0][self.tag_to_idx_mapper[self.start_tag]] = 0.
        last_step_values = pyutils.getTorchVariableFromTensor(last_step_values)
        for unary in unary_potentials:
            cur_step_values = []
            for next_tag in range(self.tagset_size):
                unary_score = unary[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transition_potentials[next_tag].view(1, -1)
                next_tag_var = last_step_values + trans_score + unary_score
                cur_step_values.append(pyutils.logSumExp(next_tag_var))
            last_step_values = torch.cat(cur_step_values).view(1, -1)
        terminal_var = last_step_values + self.transition_potentials[self.tag_to_idx_mapper[self.stop_tag]]
        alpha = pyutils.logSumExp(terminal_var)
        return alpha


    def computeSequenceScore(self, unary_potentials, tags):
        score =pyutils.getTorchVariable([0])
        start_var = pyutils.getTorchVariable( [self.tag_to_idx_mapper[self.start_tag]], typ="long" )
        tags = torch.cat([start_var, tags], dim=0)
        tags_numpy = tags.data.cpu().numpy()
        for i, unary in enumerate(unary_potentials): #unary_potentials: seqlen*tagsize
            score += self.transition_potentials[tags[i + 1], tags[i]] + unary[tags[i + 1]]
        #print "self.tag_to_idx_mapper[self.stop_tag] = ", self.tag_to_idx_mapper[self.stop_tag]
        #print "tags[-1] = ", tags[-1]
        score = score + self.transition_potentials[self.tag_to_idx_mapper[self.stop_tag], tags_numpy[-1]]
        return score


    def viterbiDecoding(self, unary_potentials):
        
        last_step_values = torch.Tensor(1, self.tagset_size).fill_(negative_inf)
        last_step_values[0][self.tag_to_idx_mapper[self.start_tag]] = 0
        last_step_values = pyutils.getTorchVariableFromTensor(last_step_values)
        best_index = []

        for unary in unary_potentials:
            cur_best_indices = []  
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = last_step_values + self.transition_potentials[next_tag]
                best_tag_idx = pyutils.getArgmax(next_tag_var)
                cur_best_indices.append(best_tag_idx)
                viterbivars_t.append(next_tag_var[0][best_tag_idx])
            last_step_values = (torch.cat(viterbivars_t) + unary).view(1, -1)
            best_index.append(cur_best_indices)

        terminal_var = last_step_values + self.transition_potentials[self.tag_to_idx_mapper[self.stop_tag]]
        best_tag_idx = pyutils.getArgmax(terminal_var)
        best_score = terminal_var[0][best_tag_idx]
        best_tag_sequence = [best_tag_idx]
        for cur_best_indices in reversed(best_index):
            best_tag_idx = cur_best_indices[best_tag_idx]
            best_tag_sequence.append(best_tag_idx)
        start = best_tag_sequence.pop()
        assert start == self.tag_to_idx_mapper[self.start_tag]  # Sanity check
        best_tag_sequence.reverse()
        return best_score, best_tag_sequence


    def getNegativeLogLikelihood(self, sequence, gold_tags, unary_potentials):
        gold_score = self.computeSequenceScore(unary_potentials, gold_tags)
        forward_score = self.computePartition(unary_potentials)
        info = forward_score.data.cpu().numpy(), gold_score.data.cpu().numpy() 
        return forward_score - gold_score, info


    def forward(self, sequence, unary_potentials): 
        score, tag_seq = self.viterbiDecoding(unary_potentials)
        return score, tag_seq


##############################################################################
##############################################################################



class SemiMarkovNeuralCRF(nn.Module):


    def __init__(self, vocab_size, tag_to_idx_mapper, start_tag, stop_tag, tag_embedding_size=32, unary_feats_size=32 ):
        
        super(SemiMarkovNeuralCRF, self).__init__()
        self.vocab_size = vocab_size # vocab_size: this is tagset vocab size
        self.tag_to_idx_mapper = tag_to_idx_mapper
        self.tagset_size = len(tag_to_idx_mapper)
        self.start_tag = start_tag
        self.stop_tag = stop_tag
        self.maxlen = 2

        self.unary_feats_size = unary_feats_size
        self.tag_embedding_size = tag_embedding_size
        self.tag_embeddings = nn.Embedding(self.tagset_size, tag_embedding_size )

        self.total_feat_size = self.unary_feats_size + 2*self.tag_embedding_size
        self.W = nn.Linear(self.total_feat_size, 1) #self.tagset_size)

        #self.transition_potentials = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))#transitions[]
        #self.transition_potentials.data[tag_to_idx_mapper[start_tag], :] = negative_inf
        #self.transition_potentials.data[:, tag_to_idx_mapper[stop_tag]] = negative_inf
        print "SemiMarkovNeuralCRF modules: ",self._modules.keys()


    def printTransitionParams(self): 
        pass


    def computePartition(self, unary_feats):

        unary_feats = [None] + unary_feats + [ pyutils.getTorchVariable(np.zeros(self.unary_feats_size)).view(1,-1) ] 
        seq_len = len(unary_feats) 

        last_step_values = [torch.Tensor(1).fill_(negative_inf) for i in range(len(self.tag_to_idx_mapper))]
        last_step_values[self.tag_to_idx_mapper[self.start_tag]][0] = 0
        last_step_values = [ pyutils.getTorchVariableFromTensor(vals) for vals in last_step_values ]
        all_step_values = []
        all_step_values.append(last_step_values)
        
        step = 0
        maxlen = self.maxlen

        for step in range(1, seq_len):
            #print "------- step = ", step
            cur_step_values = []
            for cur_tag in range(self.tagset_size):
                cur_tag_vals = []
                cur_step_tag_value =  pyutils.getTorchVariableFromTensor( torch.Tensor(1).fill_(negative_inf) )
                best_index = (-1,1)
                best_score =  pyutils.getTorchVariable(np.array([negative_inf]), typ="float")  #negative_inf
                if step==(seq_len-1):
                    if cur_tag!=self.tag_to_idx_mapper[self.stop_tag]:
                        continue
                for prev_tag in range(self.tagset_size):
                    for l in range(1,min(maxlen+1,step+1)):  
                        score =  all_step_values[step-l][prev_tag] 
                        #print "score prev = ", score
                        prev_tag_var = pyutils.getTorchVariable( np.array([prev_tag], dtype=np.int), typ="long") 
                        cur_tag_var =  pyutils.getTorchVariable( np.array([cur_tag], dtype=np.int), typ="long") 
                        score += self._computePotential(unary_feats, self.tag_embeddings(prev_tag_var), self.tag_embeddings(cur_tag_var) , step-l+1, step) 
                            # l=1 means step,step -> segment has only one word
                        cur_tag_vals.append(score.view(1,-1))
                #print "cur_tag_vals = ", len(cur_tag_vals), cur_tag_vals[0].data.shape
                cur_tag_vals = torch.cat(cur_tag_vals, dim=1)
                #print "cur_tag_vals = ", len(cur_tag_vals), cur_tag_vals[0].data.shape
                cur_step_values.append(pyutils.logSumExp(cur_tag_vals))

            all_step_values.append(cur_step_values)

        alpha =  all_step_values[-1][0] # only one value is sstored at last step #[self.tag_to_idx_mapper[self.stop_tag]] #pyutils.logSumExp(all_step_values[-1])
        return alpha


    def computeSequenceScore(self, unary_feats, tags_segments):

        # tags_segments: [ (0,0,O), (1,3,N), (4,4,O), .. ] # with O,N,.. etc replaced by index instead of string
        #print ":: ", tags_segments
        #print ":: ", len(unary_feats), len(tags_segments)
        #assert len(unary_feats) == len(tags_segments)
        unary_feats = unary_feats + [ pyutils.getTorchVariable(np.zeros(self.unary_feats_size)).view(1,-1) ]  #Note: mnot adding none in beginnning. Main thing is to make sure unary_feats get indexed correctly when calling _computePotential
        seq_len = len(unary_feats) 
        tags_segments = tags_segments + [(seq_len-1,seq_len-1,self.tag_to_idx_mapper[self.stop_tag]) ]

        score = pyutils.getTorchVariable([0])
        maxlen = self.maxlen
        prev_tag = self.tag_to_idx_mapper[self.start_tag]
        
        for i,vals in enumerate(tags_segments):
            start_idx, end_idx, tag = vals
            prev_tag_var = pyutils.getTorchVariable( np.array([prev_tag], dtype=np.int), typ="long") 
            cur_tag_var =  pyutils.getTorchVariable( np.array([tag], dtype=np.int), typ="long") 
            score =  score + self._computePotential(unary_feats, self.tag_embeddings(prev_tag_var), self.tag_embeddings(cur_tag_var), start_idx, end_idx)
            prev_tag = tag
            #print "i,score,tag, start, end: ",i,score.data.cpu().numpy(),tag, start_idx, end_idx
        return score


    def _computePotential(self, unary_feats, prev_tag_feats, cur_tag_feats, start_idx, end_idx):
        feats = unary_feats[start_idx] 
        for idx in range(start_idx+1,end_idx+1):
            feats+=unary_feats[idx]
        feats = feats / (end_idx + 1 - start_idx)
        #print "feats: ",feats.data.shape, prev_tag_feats.data.shape, cur_tag_feats.data.shape
        feats = torch.cat([feats,prev_tag_feats,cur_tag_feats], dim=1)
        # cal also add the legnth feature
        score = F.relu(self.W(feats).view(-1)) # 1,1 -> 1
        #print "score = ", score
        return score



    def viterbiDecoding(self, unary_feats):
        
        unary_feats = [None] + unary_feats + [ pyutils.getTorchVariable(np.zeros(self.unary_feats_size)).view(1,-1) ] # unary_feats must include extra feats in beginning and end for start and end repsectiley
        seq_len = len(unary_feats) 
        print ":len(unary_feats)  ", len(unary_feats) 

        #last_step_values = [torch.Tensor(1).fill_(negative_inf) for i in range(len(self.tag_to_idx_mapper))]
        last_step_values = [negative_inf for i in range(len(self.tag_to_idx_mapper))]
        last_step_values[self.tag_to_idx_mapper[self.start_tag]] = 0
        #last_step_values = [ pyutils.getTorchVariableFromTensor(vals) for vals in last_step_values ]
        
        all_step_values = []
        all_step_values.append(last_step_values)
        
        all_best_index = []
        all_best_index.append(np.zeros(len(self.tag_to_idx_mapper)))
        step = 0
        maxlen = self.maxlen

        #print " ------- step=0 "
        #print "cur_step_values: ", last_step_values #[val.data.numpy()[0] for val in last_step_values]


        for step in range(1, seq_len):
            #print "------- step = ", step
            cur_step_values = []
            cur_best_indices = []
            for cur_tag in range(self.tagset_size):
                #cur_step_tag_value =  pyutils.getTorchVariableFromTensor( torch.Tensor(1).fill_(negative_inf) )
                best_index = (-1,1)
                best_score =  negative_inf # pyutils.getTorchVariable(np.array([negative_inf]), typ="float")  #negative_inf
                for prev_tag in range(self.tagset_size):
                    if step==(seq_len-1):
                        if cur_tag!=self.tag_to_idx_mapper[self.stop_tag]:
                            continue
                    for l in range(1,min(maxlen+1,step+1)):  
                        score =  all_step_values[step-l][prev_tag] 
                        #print "score prev = ", score
                        prev_tag_var = pyutils.getTorchVariable( np.array([prev_tag], dtype=np.int), typ="long") 
                        cur_tag_var =  pyutils.getTorchVariable( np.array([cur_tag], dtype=np.int), typ="long") 
                        score += self._computePotential(unary_feats, self.tag_embeddings(prev_tag_var), self.tag_embeddings(cur_tag_var) , step-l+1, step).data.cpu().numpy()[0]
                            # l=1 means step,step -> segment has only one word
                        if score>best_score: #.data.cpu().numpy()[0]>=best_score.data.cpu().numpy()[0]:
                            best_score = score#.data.cpu().numpy()[0]
                            best_index = (prev_tag,l)
                cur_best_indices.append(best_index)
                cur_step_values.append(best_score)
            #print "cur_step_values: ",  cur_step_values
            #print "cur_best_indices: ", cur_best_indices
            all_step_values.append(cur_step_values)
            all_best_index.append(cur_best_indices)

        #print "** all_best_index = ", all_best_index
        best_tag_idx = (self.tag_to_idx_mapper[self.stop_tag],1)
        best_score = all_step_values[-1][best_tag_idx[0]]
        best_tag_sequence = [best_tag_idx]
        #for i in reversed(range(0,seq_len-1)):
        i = seq_len-1 # stop tag
        #print "i,best_tag_idx = ", i, best_tag_idx
        i = i - best_tag_idx[1] # stop tag only has 1 length
        while i>0:
            best_tag_idx = all_best_index[i][best_tag_idx[0]]
            #print "i,best_tag_idx = ", i, best_tag_idx
            i = i - best_tag_idx[1]
            best_tag_sequence.append(best_tag_idx)
        start = best_tag_sequence.pop()
        assert start[0] == self.tag_to_idx_mapper[self.start_tag]  # Sanity check
        best_tag_sequence.reverse()
        return best_score, best_tag_sequence


    '''
        for unary in unary_feats:
            cur_best_indices = []  
            viterbivars_t = []
            for prev_tag in range(self.tagset_size):
                next_tag_predictions = self._computePotential(unary_feats,self.tag_embeddings(prev_tag), step-1, step)

            step+=1

            for next_tag in range(self.tagset_size):
                next_tag_var = last_step_values + self.transition_potentials[next_tag]
                best_tag_idx = pyutils.getArgmax(next_tag_var)
                cur_best_indices.append(best_tag_idx)
                viterbivars_t.append(next_tag_var[0][best_tag_idx])
            last_step_values = (torch.cat(viterbivars_t) + unary).view(1, -1)
            all_step_values.append(last_step_values)
            best_index.append(cur_best_indices)

        terminal_var = last_step_values + self.transition_potentials[self.tag_to_idx_mapper[self.stop_tag]]
        best_tag_idx = pyutils.getArgmax(terminal_var)
        best_score = terminal_var[0][best_tag_idx]
        best_tag_sequence = [best_tag_idx]
        for cur_best_indices in reversed(best_index):
            best_tag_idx = cur_best_indices[best_tag_idx]
            best_tag_sequence.append(best_tag_idx)
        start = best_tag_sequence.pop()
        assert start == self.tag_to_idx_mapper[self.start_tag]  # Sanity check
        best_tag_sequence.reverse()
        return best_score, best_tag_sequence
    '''


    def getNegativeLogLikelihood(self, sequence, gold_tags, unary_feats):
        gold_tags = gold_tags[0]
        gold_score = self.computeSequenceScore(unary_feats, gold_tags)
        #print "gold_score = ", gold_score
        forward_score = self.computePartition(unary_feats)
        info = forward_score.data.cpu().numpy(), gold_score.data.cpu().numpy() 
        return forward_score - gold_score, info


    def forward(self, sequence, unary_feats): 
        score, tag_seq = self.viterbiDecoding(unary_feats)
        return score, tag_seq

