#pytorch
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

def getTorchVariable(vals, typ="float", volatile=False):
    if typ=="float":
        tensor = torch.FloatTensor(vals)
    elif typ=="long":
        tensor = torch.LongTensor(vals)
    if torch.cuda.is_available():
        tensor=tensor.cuda()
    return autograd.Variable(tensor, volatile=volatile)


def toNumpy(var):
    return var.data.cpu().numpy()


def getArgmax(vals):
    maxval, maxval_idx = torch.max(vals, 1)
    return toNumpy(maxval_idx)[0]


def logSumExp(vals):
    max_score = vals[0, getArgmax(vals)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vals.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vals - max_score_broadcast)))


