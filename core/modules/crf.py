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
        forward_score = self.computePartition(unary_potentials)
        gold_score = self.computeSequenceScore(unary_potentials, gold_tags)
        info = forward_score.data.cpu().numpy(), gold_score.data.cpu().numpy() 
        return forward_score - gold_score, info


    def forward(self, sequence, unary_potentials): 
        score, tag_seq = self.viterbiDecoding(unary_potentials)
        return score, tag_seq
