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
import pytorchnlp

#########################

class RNNTagger(nn.Module):

    def __init__(self, params, model_type="rnn"):
        super(RNNTagger,self).__init__()
        self.params = params
        self.model_type = model_type
        if model_type=="rnn":
            self.rnn_model = RNNModel(vocab_size=params.vocab_size, emb_size=params.emb_size, hidden_size=params.hidden_size) #, rnn_type=params.rnn_type, share_embeddings=params.share_embeddings)
            self.W = nn.Linear(params.hidden_size, params.tag_size)
        elif model_type=="birnn":
            self.rnn_model = BiRNNModel(vocab_size=params.vocab_size, emb_size=params.emb_size, hidden_size=params.hidden_size) #, rnn_type=params.rnn_type, share_embeddings=params.share_embeddings)
            self.W = nn.Linear(2*params.hidden_size, params.tag_size)
        print "RNNTagger: ",self._modules.keys()


    def forward(self, inp, gt_output, mask=None, loss_function=None, mode="train", get_loss=True):

        # gt_output: batch_size, seqlen

        gt_output = np.array(gt_output)
        hidden_size = self.params.hidden_size
        batch_size = gt_output.shape[0]
        seq_length = gt_output.shape[1]

        encoder_outputs, enc_hidden = self.rnn_model(inp, mode=mode) # encoder_outputs: seqlen, batch_size, hidden_size

        loss = 0.0
        predictions = [[] for i in range(batch_size)]
        mask = np.transpose(mask)  # mask: seqlen, batch_size
        gt_output = np.transpose(gt_output) # gt_output: seqlen, batch_size

        for step,(encoder_output_step,gt_output_step) in enumerate(zip(encoder_outputs, gt_output)):
            softmax_output = F.log_softmax(self.W(encoder_output_step))
            gt_output_step = pytorchnlp.pytorch_utils.nn.getTorchVariable(gt_output_step, typ="long")
            #print "softmax_output = ", softmax_output
            #print "gt_output_step = ", gt_output_step
            #print "loss_function(softmax_output,gt_output_step) = ", loss_function(softmax_output,gt_output_step).data
            #print "manualCalculation(softmax_output,gt_output_step) = ", manualCalculation(softmax_output.data.cpu().numpy(),gt_output_step.data.cpu().numpy())
            if get_loss:
                loss+=loss_function(softmax_output,gt_output_step)
            if mode=="decode":
                cur_step_predictions = softmax_output.data.cpu().numpy() # cur_step_predictions: batch, vocab
                cur_step_predictions = np.argmax(cur_step_predictions, axis=1) # batch
                #print "cur_step_predictions = ", cur_step_predictions
                #print "mask[step] = ", mask[step]
                for i,(cur_step_predictions_i,mask_step_i) in enumerate(zip(cur_step_predictions, mask[step])):
                    if mask_step_i==1:
                        predictions[i].append(cur_step_predictions_i)

        if get_loss:
            return loss, predictions
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



