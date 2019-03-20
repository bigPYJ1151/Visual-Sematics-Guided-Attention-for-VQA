import torch.nn as nn
import yaml
from addict import Dict

def conv_ben_relu(in_planes, out_planes, kernal, stride, pad, bn=True, relu=False, bias=False):
    mlist = []
    mlist.append(nn.Conv2d(in_planes, out_planes, kernal, stride=stride, padding=pad, bias=bias))
    
    if bn:
        mlist.append(nn.BatchNorm2d(out_planes))

    if relu:
        mlist.append(nn.ReLU(inplace=True))

    return nn.Sequential(*mlist)    
s = Dict(yaml.load(open('config.yaml')))
print(1/2)