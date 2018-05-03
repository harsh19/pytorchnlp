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
from PIL import Image, ImageFilter
import numpy as np
import random
import math

# others
from pytorchnlp.core.modules.layers import * 
from pytorchnlp.core.modules.crf import * 
import pytorchnlp


#########################

class BiRNNCRF(nn.Module):

    def __init__(self, params, tag_to_ix, model_type="rnn"):
        super(BiRNNCRF,self).__init__()
        self.params = params
        self.model_type = model_type
        self.rnn_model = BiRNNModel(vocab_size=params.vocab_size, emb_size=params.emb_size, hidden_size=params.hidden_size) #, rnn_type=params.rnn_type, share_embeddings=params.share_embeddings)
        self.W = nn.Linear(2*params.hidden_size, params.tag_size)
        self.neural_crf = NeuralCRF(params.tag_size, tag_to_ix, start_tag=params.start_tag, stop_tag=params.stop_tag)
        print "BiRNNCRF: ",self._modules.keys()


    def forward(self, inp, gt_output, mask=None, loss_function=None, mode="train", get_loss=True, debug=False):

        # gt_output: batch_size, seqlen

        gt_output = np.array(gt_output)
        hidden_size = self.params.hidden_size
        batch_size = gt_output.shape[0]
        seq_length = gt_output.shape[1]

        encoder_outputs, enc_hidden = self.rnn_model(inp, mode=mode) # encoder_outputs: seqlen, batch_size, hidden_size

        loss = 0.0
        predictions = [[] for i in range(batch_size)]
        mask = np.transpose(mask)  # mask: seqlen, batch_size
        
        all_outputs = [self.W(encoder_output_step).view(-1) for encoder_output_step in encoder_outputs]
        assert batch_size==1
        # all_outputs: seqlength, tag_size

        gt_output = pytorchnlp.pytorch_utils.nn.getTorchVariable(gt_output, typ="long").view(-1) # batchsize*seqlen -> seqlen [since batchsize=1]

        if mode=="train":
            loss, info = self.neural_crf.getNegativeLogLikelihood(inp, gt_output, all_outputs)
            fscore, goldscore = info
            if debug:
                print "fscore(z) = ", fscore, "|| goldscore =", goldscore
                self.neural_crf.printTransitionParams()
            return loss

        elif mode=="decode":
            score, predictions = self.neural_crf(inp, all_outputs)
            predictions = [predictions] # batch of size 1
            if get_loss:
                return score, predictions
            return predictions



#######################################

def manualCalculation(preds, gts):
    loss = 0.0
    for i,gtt in enumerate(gts):
        if gtt==-1:
            continue
        loss+=preds[i][gtt]
    return loss


'''
encoder_outputs = torch.cat(encoder_outputs, dim=2)
# encoder_outputs: seqlen, batch_size, hidden_size
print "encoder_outputs = ", encoder_outputs.data.shape
encoder_outputs = encoder_outputs.permute([1,0,2])
# encoder_outputs: batch_size, seqlen, hidden_size

softmax_outputs = self.W(encoder_outputs.view(-1)).view(batch_size*seq_length,-1)
# softmax_outputs: batch_size*seqlen, vocab_size

gt_output = gt_output.reshape(-1)
# gt_output: batch_size*seqlen
gt_output = pytorchnlp.pytorch_utils.nn.getTorchVariable(gt_output, typ="long")

loss = F.log_softmax(softmax_outputs, gt_output)
'''



