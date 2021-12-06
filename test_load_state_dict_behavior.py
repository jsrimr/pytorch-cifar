from os import name
import torch
import torch.nn as nn


def convert_w(l):
    l.weight.data = torch.zeros_like(l.weight)

def convert_w_with_load_state_dict(l):
    from collections import OrderedDict
    
    new_dict = OrderedDict()
    new_dict['weight'] = torch.zeros_like(l.weight)
    l.load_state_dict(new_dict)


if __name__ == "__main__":
    l = nn.Conv2d(3,10, 1, bias=False)
    print("original", l.weight)

    convert_w(l)
    print("after converting", l.weight)

    convert_w_with_load_state_dict(l)
    print("after converting with load_state_dict", l.weight)

