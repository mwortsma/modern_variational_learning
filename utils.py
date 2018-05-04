""" Utility and helper functions for AIR """

import torch
from torch.autograd import Variable

# Convert to variable which works if GPU
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

