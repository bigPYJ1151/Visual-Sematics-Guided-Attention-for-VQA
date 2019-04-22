
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
import json
from addict import Dict 
from tqdm import tqdm

CONFIG = Dict(yaml.load(open('config.yaml'))['VQA'])
CONFIG_Bi = Dict(yaml.load(open('config.yaml'))['BISENET'])

@click.command()
@click.option('--device', default='0')
@click.option('--epoch', default='49')
def main(device, epoch):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    torch.backends.cudnn.benchmark = True

    test_data = VQA(CONFIG.DATASET.COCO, CONFIG.DATASET.VQA, CONFIG.DATASET.COCO_PROCESSED,
                    CONFIG.DATASET.VOCAB, 2)
    test_loader = DataLoader(test_data, batch_size=CONFIG.DATALOADER.BATCH_SIZE.TRAIN,
                            shuffle=True, num_workers=CONFIG.DATALOADER.WORKERS, pin_memory=True, collate_fn=collate_fn)
    print("Dataset Ready.")

    record = torch.load(os.path.join(CONFIG.SOLVER.SAVE_PATH, 'VQA_Epoch_{}.pth'.format(epoch)))
    model_state = record['model']
    vqa_model = nn.DataParallel(VQA_Model(train_data.num_tokens, CONFIG.DATASET.VOCAB_NUM, CONFIG_Bi.DATASET.CLASS_NUM).cuda())
    vqa_model.load_state_dict(model_state)
    bise_model = nn.DataParallel(BiSeNet(CONFIG_Bi.DATASET.CLASS_NUM, CONFIG_Bi.DATASET.IGNORE_LABEL).cuda())
    bise_state = torch.load(os.path.join('model', CONFIG.MODEL.BISENET))
    bise_model.load_state_dict(bise_state['model'])
    pool = nn.DataParallel(nn.AdaptiveMaxPool2d(14).cuda())
    resnet = nn.DataParallel(ResNet152().cuda())
    vqa_model.eval()
    bise_model.eval()
    pool.eval()
    resnet.eval()
    print('Model ready.')

    result = []
    with torch.no_grad():
        for qlen, q, v, item in tqdm(test_loader):
            qlen = qlen.cuda()
            q = q.cuda()
            v = v.cuda()

            l = pool(bise_model(v.detach()))
            v = resnet(v.detach())

            score =  vqa_model(v.detach(), l.detach(), q, qlen)
            a_index = score.argmax(dim=1).detach().cpu()
            for a, i in zip(list(a_index), list(item)):
                ans = {
                    "question_id":int(test_data.questionids[int(i)]),
                    "answer":test_data.index_to_answer[int(a)]
                }
                result.append(ans)

    with open("record/vqa_OpenEnded_mscoco_test2015_bigpyj_results.json", 'w') as fd:
        json.dump(result, fd)

def collate_fn(batch):
    batch.sort(key = lambda x: x[0], reverse = True)
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == "__main__":
    main()      

