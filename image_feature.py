
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.BiSe_res18 import BiSeNet
from model.ResNet152 import ResNet152
from data import COCO
import click
import os
import yaml
from addict import Dict 
import h5py as h5 
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

CONFIG_VQA = Dict(yaml.load(open('config.yaml'))['VQA'])
CONFIG_BIS = Dict(yaml.load(open('config.yaml'))['BISENET'])

@click.command()
@click.option('--device', default='0')
def main(device):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    torch.backends.cudnn.benchmark = True

    result = torch.load(os.path.join('model', 'bisenet_49.pth'))

    seg_model = nn.DataParallel(BiSeNet(CONFIG_BIS.DATASET.CLASS_NUM, CONFIG_BIS.DATASET.IGNORE_LABEL).cuda())
    seg_model.load_state_dict(result['model'])
    seg_model.eval()
    resnet = nn.DataParallel(ResNet152().cuda())
    resnet.eval()
    print('Model Ready.')

    for split, name in zip(['train2014', 'val2014'], ['train_image', 'val_image']):
        loader = DataLoader(COCO(CONFIG_VQA.DATASET.COCO, split), batch_size=CONFIG_BIS.DATALOADER.BATCH_SIZE.TEST,
                        pin_memory=True, num_workers=CONFIG_BIS.DATALOADER.WORKERS, shuffle=False)
        feature_shape = (len(loader.dataset), 2048, 14, 14)
        semantic_shape = (len(loader.dataset), 448, 448)

        with h5.File(CONFIG_VQA.DATASET.COCO_PROCESSED, libver='latest') as f:
            features = f.create_dataset(name+'_feature', shape=feature_shape, dtype='float16')
            semantics = f.create_dataset(name+'_semantic', shape=semantic_shape, dtype='int8')

            with torch.no_grad():
                i = 0
                for image in tqdm(loader):
                    image = image.cuda()
                    feature = resnet(image).detach().cpu().numpy().astype(np.float16)
                    score = seg_model(image)
                    label = score.argmax(dim=1, keepdim=True).squeeze().cpu().numpy().astype(np.int32)
                    
                    features[i:(i+image.size(0))] = feature
                    semantics[i:(i+image.size(0))] = label

                    i += 1

if __name__ == "__main__":
    main()