import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from data import COCO
from model.BiSe_res18 import BiSeNet
import yaml
import os
import click
from addict import Dict 
from tqdm import tqdm

CONFIG = Dict(yaml.load(open('config.yaml'))['BISENET'])
total_iter = 0
current_iter = 0

@click.command()
@click.option('--device', default='0')
def main(device):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    torch.backends.cudnn.benchmark = True
    
    train_data = COCO(CONFIG.DATASET.COCO, 'train2017', req_label=True, req_augment=True, scales=CONFIG.DATASET.SCALES, flip=True)
    val_data = COCO(CONFIG.DATASET.COCO, 'val2017', req_label=True)
    train_loader = DataLoader(train_data, batch_size=CONFIG.DATALOADER.BATCH_SIZE.TRAIN, drop_last=True,
                            shuffle=True, num_workers=CONFIG.DATALOADER.WORKERS, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG.DATALOADER.BATCH_SIZE.TRAIN, drop_last=True,
                            shuffle=True, num_workers=CONFIG.DATALOADER.WORKERS, pin_memory=True)
    global total_iter
    total_iter = float(int(CONFIG.SOLVER.EPOCHS * len(train_data) / CONFIG.DATALOADER.BATCH_SIZE.TRAIN))
    print("Dataset Ready.")

    seg_model = nn.DataParallel(BiSeNet(CONFIG.DATASET.CLASS_NUM, CONFIG.DATASET.IGNORE_LABEL).cuda())
    optimizer = torch.optim.SGD(seg_model.parameters(), lr=CONFIG.SOLVER.INITIAL_LR, momentum=CONFIG.SOLVER.MOMENTUM, 
                                weight_decay=CONFIG.SOLVER.WEIGHT_DECAY, nesterov=True)
    recorder = {"Train":{'loss':[], 'PA':[], 'MA':[], 'MI':[]},
                "Val":{'loss':[], 'PA':[], 'MA':[], 'MI':[]}}
    print("Model Ready.")

    for epoch in range(CONFIG.SOLVER.EPOCHS):
        Epoch_Step(seg_model, train_loader, optimizer, epoch, recorder)
        Epoch_Step(seg_model, val_loader, optimizer, epoch, recorder, Train=False)

        name = "Epoch_{:d}".format(epoch)
        results = {
            'model':seg_model.state_dict(),
            'recorder':recorder
        }
        torch.save(results, os.path.join(CONFIG.SOLVER.SAVE_PATH, 'bisenet_{}.pth'.format(name)))

def Epoch_Step(target_model, loader, optimizer, epoch, recorder, Train=True):
    if Train:
        target_model.train()
        mode = 'Train'
    else:
        target_model.eval()
        mode = 'Val'

    loss_window = window()
    PA_window = window()
    MA_window = window()
    MI_window = window()
    fmt = '{:.4f}'.format
    tloader = tqdm(loader, desc='{}_Epoch:{:03d}'.format(mode, epoch), ncols=0)
    np.seterr(divide='ignore', invalid='ignore')

    for image, label, _ in tloader:
        image = image.cuda()
        label = label.cuda()

        loss, score = target_model(image, label)
        loss = loss.mean()

        loss_window.update(loss.detach().cpu().numpy())
        ans = Model_Score(label.detach().cpu().numpy(), score.argmax(dim=1, keepdim=True).cpu().numpy(), CONFIG.DATASET.CLASS_NUM)
        PA_window.update(ans["Pixel Accuracy"])
        MA_window.update(ans["Mean Accuracy"])
        MI_window.update(ans["Mean IoU"])

        optimizer.zero_grad()
        loss.backward()

        if Train:
            global current_iter
            lr_update(optimizer)
            current_iter += 1
            optimizer.step()

        tloader.set_postfix(loss=fmt(loss_window.value), pix_acc=fmt(PA_window.value))

    recorder[mode]['loss'].append(loss_window.value)
    recorder[mode]['PA'].append(PA_window.value)
    recorder[mode]['MA'].append(MA_window.value)
    recorder[mode]['MI'].append(MI_window.value)

def lr_update(optimizer):
    lr = CONFIG.SOLVER.INITIAL_LR * (1 - current_iter / total_iter) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class window:

    def __init__(self):
        self.sum = 0
        self.iter_num = 0
    
    def update(self, value):
        self.sum += value
        self.iter_num += 1
    
    @property
    def value(self):
        return self.sum / self.iter_num

def Model_Score(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

if __name__ == "__main__":
    main()