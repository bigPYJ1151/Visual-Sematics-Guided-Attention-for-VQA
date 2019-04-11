
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from data import VQA
from model.VQA import VQA_Model
from model.BiSe_res18 import BiSeNet
from model.ResNet152 import ResNet152
import yaml
import os
import click
from addict import Dict 
from tqdm import tqdm

CONFIG = Dict(yaml.load(open('config.yaml'))['VQA'])
CONFIG_Bi = Dict(yaml.load(open('config.yaml'))['BISENET'])
total_iter = 0
current_iter = 0

@click.command()
@click.option('--device', default='0')
def main(device):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    torch.backends.cudnn.benchmark = True
    
    train_data = VQA(CONFIG.DATASET.COCO, CONFIG.DATASET.VQA, CONFIG.DATASET.COCO_PROCESSED,
                    CONFIG.DATASET.VOCAB, 0)
    val_data = VQA(CONFIG.DATASET.COCO, CONFIG.DATASET.VQA, CONFIG.DATASET.COCO_PROCESSED,
                    CONFIG.DATASET.VOCAB, 1)
    train_loader = DataLoader(train_data, batch_size=CONFIG.DATALOADER.BATCH_SIZE.TRAIN, 
                            shuffle=True, num_workers=CONFIG.DATALOADER.WORKERS, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=CONFIG.DATALOADER.BATCH_SIZE.TRAIN,
                            shuffle=True, num_workers=CONFIG.DATALOADER.WORKERS, pin_memory=True, collate_fn=collate_fn)
    global total_iter
    total_iter = float(int(CONFIG.SOLVER.EPOCHS * len(train_data) / CONFIG.DATALOADER.BATCH_SIZE.TRAIN))
    print("Dataset Ready.")

    target_model = nn.DataParallel(VQA_Model(train_data.num_tokens, CONFIG.DATASET.VOCAB_NUM, CONFIG_Bi.DATASET.CLASS_NUM).cuda())
    bise_model = nn.DataParallel(BiSeNet(CONFIG_Bi.DATASET.CLASS_NUM, CONFIG_Bi.DATASET.IGNORE_LABEL).cuda())
    bise_state = torch.load(os.path.join('model', CONFIG.MODEL.BISENET))
    bise_model.load_state_dict(bise_state['model'])
    resnet = nn.DataParallel(ResNet152().cuda())
    #optimizer = torch.optim.SGD(seg_model.parameters(), lr=CONFIG.SOLVER.INITIAL_LR, momentum=CONFIG.SOLVER.MOMENTUM, 
     #                           weight_decay=CONFIG.SOLVER.WEIGHT_DECAY, nesterov=True)
    optimizer = torch.optim.Adam(target_model.parameters())
    recorder = {"Train":{'loss':[], 'acc':[]},
                "Val":{'loss':[], 'acc':[]}}
    print("Model Ready.")

    for epoch in range(CONFIG.SOLVER.EPOCHS):
        Epoch_Step(target_model, bise_model, ResNet152, train_loader, optimizer, epoch, recorder)
        Epoch_Step(target_model, bise_model, ResNet152, val_loader, optimizer, epoch, recorder, Train=False)

        name = "Epoch_{:d}".format(epoch)
        results = {
            'model':target_model.state_dict(),
            'recorder':recorder
        }
        torch.save(results, os.path.join(CONFIG.SOLVER.SAVE_PATH, 'VQA_{}.pth'.format(name)))

def Epoch_Step(target_model, seg_model, res_model, loader, optimizer, epoch, recorder, Train=True):
    if Train:
        target_model.train()
        mode = 'Train'
    else:
        target_model.eval()
        mode = 'Val'

    seg_model.eval()
    res_model.eval()
    loss_window = window()
    acc_window = window()
    fmt = '{:.4f}'.format
    log_softmax = nn.LogSoftmax(dim=1).cuda()
    tloader = tqdm(loader, desc='{}_Epoch:{:03d}'.format(mode, epoch), ncols=0)
    np.seterr(divide='ignore', invalid='ignore')

    for qlen, q, a, v, item in tloader:
        qlen = qlen.cuda()
        q = q.cuda()
        a = a.cuda()
        v = v.cuda()
        with torch.no_grad():
            l = seg_model(v.detach())
            v = res_model(v.detach())

        score = target_model(v.detach(), l.detach(), q, qlen)
        soft = -log_softmax(score)
        loss = (soft * a / 10).sum(dim=1).mean()
        acc = check_accuracy(score.detach(), a)

        loss_window.update(loss.detach().cpu().numpy())
        acc_window.update(acc.mean().detach().cpu().numpy())

        if Train:
            global current_iter
            lr_update2(optimizer)
            current_iter += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tloader.set_postfix(loss=fmt(loss_window.value), acc=fmt(acc_window.value))

    recorder[mode]['loss'].append(loss_window.value)
    recorder[mode]['acc'].append(acc_window.value)

def lr_update1(optimizer):
    lr = CONFIG.SOLVER.INITIAL_LR * (1 - current_iter / total_iter) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def lr_update2(optimizer):
    lr = CONFIG.SOLVER.INITIAL_LR * 0.5 ** (float(current_iter)/(total_iter))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def check_accuracy(predict, real):
    index = predict.argmax(dim = 1, keepdim = True)
    current_num = real.gather(dim=1, index=index)
    return (current_num * 0.3).clamp(max=1)

def collate_fn(batch):
    batch.sort(key = lambda x: x[0], reverse = True)
    return torch.utils.data.dataloader.default_collate(batch)

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

if __name__ == "__main__":
    main()
