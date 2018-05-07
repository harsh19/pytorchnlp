import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from pytorchnlp.pytorch_utils import nn as pyutils


class NeuralCRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, start_tag, stop_tag ):

        super(NeuralCRF, self).__init__()
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.start_tag = start_tag
        self.stop_tag = stop_tag

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag 
        self.transitions.data[tag_to_ix[start_tag], :] = -10000
        self.transitions.data[:, tag_to_ix[stop_tag]] = -10000
        print "NeuralCRF modules: ",self._modules.keys()


    def printTransitionParams(self):
        print "self.transitions.data.cpu().numpy(): "
        print "\t",
        tagset = self.tag_to_ix.keys()
        for tag in tagset:
            print tag,"\t",
        print ""
        for i,row in  enumerate(self.transitions.data.cpu().numpy()):
            print tagset[i],"\t",
            for col in row:
                print "%.2f\t" % col,
            print ""


    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # self.start_tag has all of the score.
        init_alphas[0][self.tag_to_ix[self.start_tag]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = pyutils.getTorchVariableFromTensor(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(pyutils.logSumExp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.stop_tag]]
        alpha = pyutils.logSumExp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score =pyutils.getTorchVariable([0])
        start_var = pyutils.getTorchVariable( [self.tag_to_ix[self.start_tag]], typ="long" )
        tags = torch.cat([start_var, tags], dim=0)
        tags_numpy = tags.data.cpu().numpy()
        for i, feat in enumerate(feats): #feats: seqlen*tagsize
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        #print "self.tag_to_ix[self.stop_tag] = ", self.tag_to_ix[self.stop_tag]
        #print "tags[-1] = ", tags[-1]
        score = score + self.transitions[self.tag_to_ix[self.stop_tag], tags_numpy[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[self.start_tag]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = pyutils.getTorchVariableFromTensor(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = pyutils.getArgmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to self.stop_tag
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.stop_tag]]
        best_tag_id = pyutils.getArgmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.start_tag]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, gold_tags, feats):
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, gold_tags)
        info = forward_score.data.cpu().numpy(), gold_score.data.cpu().numpy() 
        return forward_score - gold_score, info

    def forward(self, sentence, feats):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq
