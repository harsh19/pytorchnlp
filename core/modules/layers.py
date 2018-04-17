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
from pytorchnlp import pytorch_utils
import vgg

#######################################
class VGGNetFeats(torch.nn.Module):

    def __init__(self, vgg_is_fixed, retain_upto=4):
        super(VGGNetFeats,self).__init__()
        self.vgg_model = vgg.vgg16(pretrained=True, retain_upto=retain_upto)
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        self.vgg_is_fixed = vgg_is_fixed
        self.retain_upto = retain_upto
        print "VGGNetFeats: ",self._modules.keys()
        #for param in self.parameters():
        #    print(type(param.data), param.size())
        #print "="*10

    def _loadImage(self, img_path, perform_masking=False, pixels_diff=None):
        img = Image.open(img_path)
        img = self.to_tensor(img)
        
        pixels1_masked = None
        #print img
        ## masking
        #print "perform_masking = ", perform_masking
        if perform_masking:
            #print pixels_diff
            mask = torch.Tensor( pixels_diff )
            mask = mask.permute(2,0,1)
            #print mask.shape, img.shape
            img = img * mask
	#delete(mask)
        img_norm = self.normalizer(img)
        img_norm_batch = img_norm.unsqueeze(0)
        #print(img_norm_batch.size())
        return img_norm_batch

    def _getVGGFeatsFromNormBatch(self, img_norm_batch):
        inp_var = torch.autograd.Variable(img_norm_batch)
        if torch.cuda.is_available():
            inp_var = inp_var.cuda()
        feats = self.vgg_model( inp_var )
        return feats

    def forward(self, img_path, perform_masking=False, pixels_diff=None):
        #print "img_path = ",img_path
        img_norm_batch = self._loadImage(img_path, perform_masking=perform_masking, pixels_diff=pixels_diff)
        feats = self._getVGGFeatsFromNormBatch(img_norm_batch) # feats: 1,512,7,7 OR 1,512,14,14
        if self.retain_upto == 4:
            feats = feats.view(512,196)
            #print "Using upto 4th layer only"
        else:
            feats = feats.view(512,49)
        feats = feats.permute(1,0)
        if self.vgg_is_fixed:
            feats = torch.autograd.Variable(feats.data) # disconecting from vgg nets so that gradients dont flow
            if torch.cuda.is_available():
                feats = feats.cuda()
        return feats

#######################################

class AttnDecoderRNN(nn.Module):

    def __init__(self, vocab_size=100, emb_size=32, hidden_size=64, encoder_feat_size=512):

        super(AttnDecoderRNN,self).__init__()
        self.vocab_size=vocab_size
        self.emb_size=emb_size
        self.hidden_size=hidden_size
        self.embeddings=nn.Embedding(self.vocab_size,self.emb_size)
        inp_size = emb_size + encoder_feat_size
        self.decoder=nn.LSTM(inp_size, self.hidden_size)
        self.enc_transform = nn.Linear(encoder_feat_size, hidden_size)
        self.hid_transform = nn.Linear(hidden_size, hidden_size)
        print "AttnDecoderRNN: ",self._modules.keys()
        #for param in self.parameters():
        #    print(type(param.data), param.size())
        #print "="*10


    def _getAllZeros(self): # dummy. for debugging.
        hiddenElem=torch.zeros(1,self.hidden_size) # 1,b,hidden_size
        if torch.cuda.is_available():
            hiddenElem=hiddenElem.cuda()
        hiddenElem = autograd.Variable(hiddenElem)
        return hiddenElem

    def _computeAttention(self, encoder_outputs, previous_output):
        encoder_outputs_transformed =   self.enc_transform(encoder_outputs) # B*49*hidden_size
        query = self.hid_transform(previous_output) # query: B*hidden_size
        query = query.unsqueeze(1) #(encoder_outputs_transformed.size()) # query: B*1*hidden_size
        dotProduct = torch.sum(torch.mul(encoder_outputs_transformed,query),2) #B,49
        attn_weights = F.softmax(dotProduct).unsqueeze(2) # B,49,1
        context_vector = torch.sum(attn_weights*encoder_outputs_transformed,1) # B*hidden_size
        return context_vector, attn_weights
        # context_vector: B*encoder_feat_size
        # attn_weights: B*49


    def forward(self, batch_size, current_input, encoder_outputs, previous_output, previous_hidden):
        '''
        encoder_outputs: batch_size*7*7*512. encoder_feat_size=512
        current_input: batch_size * 1
        previous_output: batch_size * hidden_size
        previous_hidden: tuple. each is batch_size * hidden_size
        '''
        encoder_outputs = encoder_outputs.view(batch_size,-1,512) # batch_size*7*7*512 -> batch_size*49*512
        current_input_embeddings = self.embeddings(current_input) # current_input_embeddings: B*embedding_size
        context_vector, attn_weights = self._computeAttention(encoder_outputs, previous_output)
        rnn_input = torch.cat([current_input_embeddings,context_vector],1).view(1,batch_size,-1)
        previous_hidden = previous_hidden[0].view(1,batch_size,-1), previous_hidden[1].view(1,batch_size,-1)
        #print "previous_hidden  = ", previous_hidden
        out,hidden=self.decoder(rnn_input, previous_hidden)
        out = out.squeeze(0) # 1,B,hideen -> B,hidden
        return out,hidden,context_vector,attn_weights



#######################################


class RNNEncoder(nn.Module):
   
    def __init__(self, vocab_size=100, emb_size=32, hidden_size=64, rnn_type='lstm', \
            embeddings_supplied=None):
        
        super(RNNEncoder,self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        if embeddings_supplied is None:
            self.embeddings = nn.Embedding(self.vocab_size,self.emb_size)
        else:
            self.embeddings = None  #TODO
        inp_size = emb_size
        self.rnn_type = rnn_type
        self.encoder = nn.LSTM(inp_size, self.hidden_size)
        print "RNNEncoder: ",self._modules.keys()


    def getZeroState(self, batch_size): 
        hidden_ele = torch.zeros(1,batch_size, self.hidden_size) # 1,b,hidden_size
        if torch.cuda.is_available():
            hidden_ele = hidden_ele.cuda()
        hidden_ele = autograd.Variable(hidden_ele)
        return hidden_ele


    def getEmbeddingsLayer(self):
        return self.embeddings


    def forward(self, batch_size, current_input, previous_output, previous_hidden, mode="train"):
        '''
        current_input: batch_size * 1
        previous_output: batch_size * hidden_size
        previous_hidden: tuple. each is batch_size * hidden_size
        '''
        current_input_embeddings = self.embeddings(current_input) # current_input_embeddings: B*embedding_size
        #print "current_input_embeddings = ", current_input_embeddings
        rnn_input = current_input_embeddings.view(1,batch_size,-1)
        previous_hidden = previous_hidden[0].view(1,batch_size,-1), previous_hidden[1].view(1,batch_size,-1)
        out,hidden=self.encoder(rnn_input, previous_hidden)
        out = out.squeeze(0) # 1,B,hideen -> B,hidden
        return out,hidden


##########################


class RNNModel(nn.Module):

    def __init__(self, vocab_size=100, emb_size=32, hidden_size=64, rnn_type='lstm', share_embeddings=False):
        
        super(RNNModel,self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        inp_size = emb_size
        self.rnn_type = rnn_type
        self.encoder = RNNEncoder(vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size, rnn_type=rnn_type, embeddings_supplied=None)
        print "RNNModel: ",self._modules.keys()


    def forward(self, inp, mode="train"):
        '''
        inp: batchsize * seqlength
        '''
        batch_size = inp.shape[0]

        all_zeros = self.encoder.getZeroState(batch_size)
        #print "all_zeros = ", all_zeros
        #print "hidden_size = ", self.hidden_size
        enc_hidden = all_zeros
        if self.rnn_type=="lstm":
            enc_hidden = all_zeros, all_zeros
        enc_out = all_zeros
        inp = inp.T # seqlen, batchsize

        encoder_outputs = []
        encoder_hidden = []

        for step,step_batch in enumerate(inp): # seqlen, batchsize
            current_input = pytorch_utils.nn.getTorchVariable(step_batch, typ="long", volatile=False) # batchsize, emb_size
            enc_out, enc_hidden = self.encoder(batch_size, current_input, enc_out, enc_hidden)
            encoder_outputs.append(enc_out.squeeze(0)) # enc_out: 1,b,hidden_size. After squeeze: b,hidden_size

        return encoder_outputs, enc_hidden # enc_hidden is last step output


#######################################



class BiRNNModel(nn.Module):
    
    def __init__(self, vocab_size=100, emb_size=32, hidden_size=64, rnn_type="lstm", share_embeddings=True):
        
        super(BiRNNModel,self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        inp_size = emb_size
        self.encoder = RNNEncoder(vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size, rnn_type=rnn_type, embeddings_supplied=None)
        embeddings_supplied = None
        if not share_embeddings:
            embeddings_supplied = self.encoder.getEmbeddingsLayer()
        self.revcoder = RNNEncoder(vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size, rnn_type=rnn_type, embeddings_supplied=embeddings_supplied)
        print "BiRNNModel: ",self._modules.keys()


    def forward(self, inp, mode="train", encoder_revcoder_comb="cat"):
        '''
        inp: batchsize * seqlength
        '''

        batch_size = inp.data.shape[0]
        inp = inp.T # seqlen, batchsize

        # encoder
        encoder_outputs = []
        all_zeros = self.getZeroState()
        enc_hidden = all_zeros
        if self.rnn_type=="lstm":
            enc_hidden = all_zeros, all_zeros
        enc_out = all_zeros
        all_current_input = []
        for step,step_batch in enumerate(inp): # seqlen, batchsize
            current_input = pytorch_utils.nn.getTorchVariable(step_batch, typ="long", volatile=False) # batchsize, emb_size
            enc_out, enc_hidden = self.encoder(batch_size, current_input, enc_out, enc_hidden)
            encoder_outputs.append(enc_out.squeeze(0)) # enc_out: 1,b,hidden_size. After squeeze: b,hidden_size
            all_current_input.append(current_input)

        #revcoder
        revcoder_outputs = []
        rev_hidden = all_zeros
        if self.rnn_type=="lstm":
            rev_hidden = all_zeros, all_zeros
        rev_out = all_zeros
        for current_input in reversed(all_current_input):
            rev_out, rev_hidden = self.revcoder(batch_size, current_input, rev_out, rev_hidden)
            revcoder_outputs.append(rev_out.squeeze(0))
        revcoder_outputs.reverse()

        #combine
        combined_outputs = None
        if encoder_revcoder_comb=="cat":
            combined_outputs = [torch.concat(x,y) for x,y in zip(encoder_outputs,revcoder_outputs)]
        else:
            print "Not suppoerted"

        return combined_outputs, (enc_hidden, rev_hidden) # enc_hidden is last step output



#######################################

class LMRNN(nn.Module):

    def __init__(self, vocab_size=100, emb_size=32, hidden_size=64):

        super(LMRNN,self).__init__()
        self.vocab_size=vocab_size
        self.emb_size=emb_size
        self.hidden_size=hidden_size
        self.embeddings=nn.Embedding(self.vocab_size,self.emb_size)
        inp_size = emb_size
        self.decoder=nn.LSTM(inp_size, self.hidden_size)
        print "LMRNN: ",self._modules.keys()
        #for param in self.parameters():
        #    print(type(param.data), param.size())
        #print "="*10

    def getEmbeddingValues(self):
        return  self.embeddings.weight.data.cpu().numpy()

    def loadVectorsFromDict(self, dct, indexer):
        #dct: {word: embedding_vector}
        for word,vals in dct.items():
            if word not in indexer:
                print "Warning: ",word," not found in indexer"
            else:
                idx = indexer[word]
                self.embeddings.weight.data[idx] = torch.FloatTensor(vals)

    def _getAllZeros(self): # dummy. for debugging.
        hiddenElem=torch.zeros(1,self.hidden_size) # 1,b,hidden_size
        if torch.cuda.is_available():
            hiddenElem=hiddenElem.cuda()
        hiddenElem = autograd.Variable(hiddenElem)
        return hiddenElem


    def forward(self, batch_size, current_input, previous_output, previous_hidden):
        '''
        current_input: batch_size * 1
        previous_output: batch_size * hidden_size
        previous_hidden: tuple. each is batch_size * hidden_size
        '''
        current_input_embeddings = self.embeddings(current_input) # current_input_embeddings: B*embedding_size
        rnn_input = current_input_embeddings.view(1,batch_size,-1)
        #print "rnn_input = ",rnn_input.data.shape
        previous_hidden = previous_hidden[0].view(1,batch_size,-1), previous_hidden[1].view(1,batch_size,-1)
        #print "previous_hidden  = ", previous_hidden[0].data.shape
        out,hidden=self.decoder(rnn_input, previous_hidden)
        out = out.squeeze(0) # 1,B,hideen -> B,hidden
        #print "* out = ", out.data.shape
        return out,hidden




