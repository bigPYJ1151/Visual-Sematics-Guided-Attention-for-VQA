
import torch
import torch.nn as nn
import yaml
from addict import Dict
import click
from model.BiSe_res18 import BiSeNet
import os

CONFIG = yaml.load(open('config.yaml'))['VQA']
print(CONFIG)