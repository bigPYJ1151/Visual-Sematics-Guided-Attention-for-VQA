
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.BiSe_res18 import BiSeNet
from model.ResNet152 import ResNet152
from data import COCO, VQA
import click
import os
import matplotlib.pyplot as plt
import yaml
from addict import Dict 
import h5py as h5 
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
CONFIG_VQA = Dict(yaml.load(open('config.yaml'))['VQA'])

VQA_data = VQA(CONFIG_VQA.DATASET.COCO, CONFIG_VQA.DATASET.VQA, 
            CONFIG_VQA.DATASET.COCO_PROCESSED, CONFIG_VQA.DATASET.VOCAB, 0)
index = 177

qlen, q, a, v, l, item = VQA_data[index]
print(len(VQA_data))
print(item)
print(qlen)
print(q)
print(a)
print(v.shape)
print(v)
print(l.shape)
print(l)
v, q, a = VQA_data.data_recover(index)
plt.figure()
plt.imshow(v)
plt.title(q + '/n' + a)
plt.show()

