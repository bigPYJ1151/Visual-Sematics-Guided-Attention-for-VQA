
import torch
import torch.nn as nn
import yaml
from addict import Dict
import click
from model.BiSe_res18 import BiSeNet
import os
import numpy as np 

s = np.array([1.2,3,4]).astype(np.float32)