
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.BiSe_res18 import BiSeNet
import click
import os
import yaml
from addict import Dict 
import h5py as h5 
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

CONFIG = Dict(yaml.load(open('config.yaml'))['BISENET'])

def image(v, name):
    plt.imsave('{}.png'.format(name), v.transpose(1,2,0))

def ans_plot(train, val, name):
    plt.rcParams['figure.figsize'] = (8, 4.944)
    plt.rcParams['savefig.dpi'] = 800
    plt.rcParams['figure.dpi'] = 800

    plt.figure()
    plt.plot(train , label = "Train")
    plt.plot(val, label = 'Val')
    plt.xlim(xmin=0)
    plt.title("{} log".format(name))
    plt.xlabel('epoches')
    plt.ylabel(name)
    plt.xticks(np.arange(0, 50))
    plt.legend()
    plt.grid()
    plt.savefig("{}_all.jpg".format(name))

@click.command()
@click.option('--device', default='0')
def main(device):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    torch.backends.cudnn.benchmark = True

    result = torch.load(os.path.join('record', 'bisenet_49.pth'))
    recorder = result['recorder']
    ans_plot(recorder['Train']['loss'], recorder['Val']['loss'], 'loss')
    ans_plot(recorder['Train']['PA'], recorder['Val']['PA'], 'Pixel acc')
    ans_plot(recorder['Train']['MA'], recorder['Val']['MA'], 'Mean acc')
    ans_plot(recorder['Train']['MI'], recorder['Val']['MI'], 'Mean IOU')

    train_data = COCO(CONFIG.DATASET.COCO, 'train2017', req_label=True)
    val_data = COCO(CONFIG.DATASET.COCO, 'val2017', req_label=True)

    seg_model = nn.DataParallel(BiSeNet(CONFIG.DATASET.CLASS_NUM, CONFIG.DATASET.IGNORE_LABEL).cuda())
    seg_model.load_state_dict(result['model'])
    seg_model.eval()
    print('Model Ready.')

    with torch.no_grad():
        for i in range(10):
            im_train, la_train = train_data[i]

            loss, score = seg_model(im_train.unsqueeze(0).cuda(), la_train.unsqueeze(0).cuda())
            pre_la_train = score.argmax(dim=1, keepdim=True).squeeze(0).cpu().numpy()
            image(pre_la_train, 'pre_train{}'.format(i))
            image(la_train, 'real_train{}'.format(i))

            im_train, la_train = val_data[i]
            loss, score = seg_model(im_train.unsqueeze(0).cuda(), la_train.unsqueeze(0).cuda())
            pre_la_train = score.argmax(dim=1, keepdim=True).squeeze(0).cpu().numpy()
            image(pre_la_train, 'pre_val{}'.format(i))
            image(la_train, 'real_val{}'.format(i))

if __name__ == "__main__":
    main()