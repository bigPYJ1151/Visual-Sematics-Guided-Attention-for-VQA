
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
    pool = nn.AdaptiveMaxPool2d(14).cuda()
    print('Model Ready.')

    for split, name in zip(['train2014', 'val2014'], ['train_image', 'val_image']):
        loader = DataLoader(COCO(CONFIG_VQA.DATASET.COCO, split), batch_size=CONFIG_BIS.DATALOADER.BATCH_SIZE.TEST,
                        pin_memory=True, num_workers=CONFIG_BIS.DATALOADER.WORKERS, shuffle=False)
        feature_shape = (len(loader.dataset), 2048, 14, 14)
        semantic_shape = (len(loader.dataset), 182, 14, 14)
        id_shape = (len(loader.dataset),)

        with h5.File(CONFIG_VQA.DATASET.COCO_PROCESSED, libver='latest') as f:
            features = f.create_dataset(name+'_feature', shape=feature_shape, dtype='float16')
            semantics = f.create_dataset(name+'_semantic', shape=semantic_shape, dtype='float16')
            ids = f.create_dataset(name+'_ids', shape=id_shape, dtype='int32')

            with torch.no_grad():
                i = 0
                for image, COCOid in tqdm(loader):
                    image = image.cuda()
                    feature = resnet(image).detach().cpu().numpy().astype(np.float16)
                    print(feature.shape)
                    score = seg_model(image)
                    print(score.size())
                    score = pool(score).detach().cpu().numpy().astype(np.float16)
                    print(score.shape)
                    
                    features[i:(i+image.size(0))] = feature
                    semantics[i:(i+image.size(0))] = score
                    ids[i:(i+image.size(0))] = COCOid.detach().numpy().astype(np.int32)

                    i += image.size(0)

if __name__ == "__main__":
    main()