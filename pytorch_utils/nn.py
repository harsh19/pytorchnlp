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
