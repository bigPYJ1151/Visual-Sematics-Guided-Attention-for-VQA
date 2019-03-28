
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
CONFIG_BIS = Dict(yaml.load(open('config.yaml'))['BISENET'])

VQA_data = VQA(CONFIG_VQA.DATASET.COCO, CONFIG_VQA.DATASET.VQA, 
            CONFIG_VQA.DATASET.COCO_PROCESSED, CONFIG_VQA.DATASET.VOCAB, 0)
index = 1822

qlen, q, a, vo, v, l, item = VQA_data[index]
im, q, a = VQA_data.data_recover(index)
vs = v.cuda().unsqueeze(0)
ls = l.cuda().unsqueeze(0)
vo = vo.cuda().unsqueeze(0) 

torch.backends.cudnn.benchmark = True
result = torch.load(os.path.join('model', CONFIG_VQA.MODEL.BISENET))
seg_model = nn.DataParallel(BiSeNet(CONFIG_BIS.DATASET.CLASS_NUM, CONFIG_BIS.DATASET.IGNORE_LABEL).cuda())
seg_model.load_state_dict(result['model'])
seg_model.eval()
resnet = nn.DataParallel(ResNet152().cuda())
resnet.eval()
pool = nn.AdaptiveMaxPool2d(14).cuda()

vc = resnet(vo).detach()
lc8 = seg_model(vo).detach()
lc = pool(lc8).detach()
print(vs-vc)
print('v_feature:', torch.max(vs-vc))
print(ls-lc)
print('l_feature:', torch.max(ls-lc))
sm8 = lc8.argmax(dim=1).detach().squeeze().cpu().numpy()
sm = lc.argmax(dim=1).detach().squeeze().cpu().numpy()
print(sm8)
print(sm)
plt.figure()
plt.imshow(im)
plt.show()