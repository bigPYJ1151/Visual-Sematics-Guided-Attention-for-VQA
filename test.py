
import torch
import torch.nn as nn
import yaml
from addict import Dict
import click
from model.BiSe_res18 import BiSeNet
import os

m = nn.DataParallel(BiSeNet(200, 1).cuda())
print(m.named_parameters)