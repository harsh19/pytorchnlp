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

    def __init__(self, params, tag_to_ix, model_type="birnn_crf"): #model_type: birnn_crf, birnn_semi_crf
        super(BiRNNCRF,self).__init__()
        self.params = params
        self.model_type = model_type
        self.rnn_model = BiRNNModel(vocab_size=params.vocab_size, emb_size=params.emb_size, hidden_size=params.hidden_size) #, rnn_type=params.rnn_type, share_embeddings=params.share_embeddings)
        if model_type=="birnn_crf":
            self.W = nn.Linear(2*params.hidden_size, params.tag_size)
            self.neural_crf = NeuralCRF(params.tag_size, tag_to_ix, start_tag=params.start_tag, stop_tag=params.stop_tag)
        else:
            self.neural_crf = SemiMarkovNeuralCRF(params.tag_size, tag_to_ix, start_tag=params.start_tag, stop_tag=params.stop_tag, unary_feats_size=2*params.hidden_size, tag_embedding_size=32 )
        print "BiRNNCRF: ",self._modules.keys()


    def forward(self, inp, gt_output, mask=None, loss_function=None, mode="train", get_loss=True, debug=False):

        # gt_output: batch_size, seqlen
        if debug:
            print "="*33

        gt_output = np.array(gt_output)
        hidden_size = self.params.hidden_size
        batch_size = gt_output.shape[0]
        seq_length = gt_output.shape[1]

        encoder_outputs, enc_hidden = self.rnn_model(inp, mode=mode) # encoder_outputs: seqlen, batch_size, hidden_size

        loss = 0.0
        predictions = [[] for i in range(batch_size)]
        mask = np.transpose(mask)  # mask: seqlen, batch_size
        
        if self.model_type=="birnn_crf":
            all_outputs = [self.W(encoder_output_step).view(-1) for encoder_output_step in encoder_outputs]
        else:
            all_outputs = encoder_outputs

        assert batch_size==1
        # all_outputs: seqlength, tag_size

        gt_output_seq = gt_output

        if self.model_type=="birnn_crf":
            gt_output = pytorchnlp.pytorch_utils.nn.getTorchVariable(gt_output, typ="long").view(-1) # batchsize*seqlen -> seqlen [since batchsize=1]
        else:
            print "gt_output = ", gt_output
            gt_output = gt_output[0]
            i = 0
            tmp = []
            while i<len(gt_output):
                start = i
                i+=1
                while i<len(gt_output) and gt_output[i-1]==gt_output[i] and gt_output[i]!=3: # 3 is tag for 'O'
                    i+=1
                end = i-1
                tmp.append((start,end,gt_output[start]))
            gt_output = [tmp]

        #mode = "decode"
        #print "gt_output = ", gt_output

        if mode=="train":
            loss, info = self.neural_crf.getNegativeLogLikelihood(inp, gt_output, all_outputs)
            fscore, goldscore = info
            if debug:
                print "fscore(z) = ", fscore, "|| goldscore =", goldscore
                self.neural_crf.printTransitionParams()
            return loss

        elif mode=="decode":
            score, predictions = self.neural_crf(inp, all_outputs)
            if self.model_type=="birnn_crf":
                predictions = [predictions]
            else:
                tmp = []
                for pred in predictions:
                    for i in range(pred[1]):
                        tmp.append(pred[0])
                #assert len(tmp) == len(gt_output_seq)
                predictions = [tmp] # batch of size 1
                if debug:
                    print "predictions = ", predictions
                    print "gt_output_seq = ", gt_output_seq
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

#######################################



