
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import numpy as np
from data import COCO
import yaml
from addict import Dict 
from tqdm import tqdm

def main():
    CONFIG = Dict(yaml.load(open('config.yaml'))['BISENET'])
    dtype = torch.float32
    device = torch.device('cuda:0')

    train_data = COCO(CONFIG.DATASET.COCO, 'train2017', req_label=True, req_augment=True, scales=CONFIG.DATASET.SCALES, flip=True)
    train_loader = DataLoader(train_data, batch_size=CONFIG.DATALOADER.BATCH_SIZE.TRAIN, 
                            shuffle=True, num_workers=CONFIG.DATALOADER.WORKERS, pin_memory=True)
    seg_model = BiSeNet(CONFIG.DATASET.CLASS_NUM)
    seg_model = seg_model.to(dtype=dtype, device=device)
    loss_layer = nn.CrossEntropyLoss(ignore_index=CONFIG.DATASET.IGNORE_LABEL).to(dtype=dtype, device=device)
    print("Model Ready.")

    recoder = {'Train':{'loss':[], 'PA':[], 'MA':[], 'MI':[]},
                'Val':{'loss':[], 'PA':[], 'MA':[], 'MI':[]}}
    lr = CONFIG.SOLVER.INITIAL_RL
    iter_num = 0
    optimizer = torch.optim.SGD(seg_model.parameters(), lr=lr, momentum=CONFIG.SOLVER.MOMENTUM, weight_decay=CONFIG.SOLVER.WEIGHT_DECAY, nesterov=True)
    torch.backends.cudnn.benchmark = True
    for i in range(3):
        seg_model.train()
        mode = 'Train'
        loss_window = []
        PA_window = []
        MA_window = []
        MI_window = []
        loader = tqdm(train_loader, desc='{}_Epoch:{:03d}'.format(mode, epoch), ncols=0)
        for image, label in loader:
            image = image.to(device=device)
            label = label.to(device=device)

            score = seg_model(image)
            loss1 = loss_layer(score[0], label)
            loss2 = loss_layer(score[1], label)
            loss3 = loss_layer(score[2], label)
            loss = loss1 + loss2 + loss3
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            
            predict = score[2].detach().argmax(dim=1, keepdim=True).cpu().numpy()
            label = label.detach().cpu().numpy()
            ans = Model_Score(label, predict, CONFIG.DATASET.CLASS_NUM)
            loss_window.append(loss.detach().cpu())
            PA_window.append(ans["Pixel Accuracy"])
            MA_window.append(ans["Mean Accuracy"])
            MI_window.append[ans["Mean IoU"]]
            fmt = '{:.4f}'.format
            loader.set_postfix(loss=fmt(np.mean(loss_window)), pix_acc=fmt(np.mean(PA_window)))
        recoder['Val']['loss'].append(np.mean(loss_window))
        recoder['Val']['PA'].append(np.mean(PA_window))
        recoder['Val']['MA'].append(np.mean(MA_window))
        recoder['Val']['MI'].append(np.mean(MI_window))


def train(s):
    CONFIG = Dict(yaml.load(open('config.yaml'))['BISENET'])
    dtype = torch.float32
    device = torch.device(s)

    train_data = COCO(CONFIG.DATASET.COCO, 'train2017', req_label=True, req_augment=True, scales=CONFIG.DATASET.SCALES, flip=True)
    val_data = COCO(CONFIG.DATASET.COCO, 'val2017', req_label=True)
    train_loader = DataLoader(train_data, batch_size=CONFIG.DATALOADER.BATCH_SIZE.TRAIN, 
                            shuffle=True, num_workers=CONFIG.DATALOADER.WORKERS, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG.DATALOADER.BATCH_SIZE.TEST, 
                            shuffle=True, num_workers=CONFIG.DATALOADER.WORKERS, pin_memory=True)
    
    print("Dataset Ready.")

    seg_model = BiSeNet(CONFIG.DATASET.CLASS_NUM)
    seg_model = seg_model.to(dtype=dtype, device=device)
    loss_layer = nn.CrossEntropyLoss(ignore_index=CONFIG.DATASET.IGNORE_LABEL).to(dtype=dtype, device=device)
    print("Model Ready.")

    recoder = {'Train':{'loss':[], 'PA':[], 'MA':[], 'MI':[]},
                'Val':{'loss':[], 'PA':[], 'MA':[], 'MI':[]}}
    total_iter = float(int(CONFIG.SOLVER.EPOCHS * len(train_data) / CONFIG.DATALOADER.BATCH_SIZE.TRAIN))
    lr = CONFIG.SOLVER.INITIAL_RL
    iter_num = 0
    optimizer = torch.optim.SGD(seg_model.parameters(), lr=lr, momentum=CONFIG.SOLVER.MOMENTUM, weight_decay=CONFIG.SOLVER.WEIGHT_DECAY, nesterov=True)
    torch.backends.cudnn.benchmark = True

    for epoch in range(CONFIG.SOLVER.EPOCHS):
        #Train
        seg_model.train()
        mode = 'Train'
        loss_window = []
        PA_window = []
        MA_window = []
        MI_window = []
        loader = tqdm(train_loader, desc='{}_Epoch:{:03d}'.format(mode, epoch), ncols=0)
        for image, label in loader:
            image = image.to(device=device)
            label = label.to(device=device)

            score = seg_model(image)
            loss1 = loss_layer(score[0], label)
            loss2 = loss_layer(score[1], label)
            loss3 = loss_layer(score[2], label)
            loss = loss1 + loss2 + loss3
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            iter_num += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (1 - iter_num / total_iter) ** 0.9

            predict = score[2].detach().argmax(dim=1, keepdim=True).cpu().numpy()
            label = label.detach().cpu().numpy()
            ans = Model_Score(label, predict, CONFIG.DATASET.CLASS_NUM)
            loss_window.append(loss.detach().cpu())
            PA_window.append(ans["Pixel Accuracy"])
            MA_window.append(ans["Mean Accuracy"])
            MI_window.append[ans["Mean IoU"]]
            fmt = '{:.4f}'.format
            loader.set_postfix(loss=fmt(np.mean(loss_window)), pix_acc=fmt(np.mean(PA_window)))
        recoder['Val']['loss'].append(np.mean(loss_window))
        recoder['Val']['PA'].append(np.mean(PA_window))
        recoder['Val']['MA'].append(np.mean(MA_window))
        recoder['Val']['MI'].append(np.mean(MI_window))
        # Train

        # Val
        seg_model.eval()
        mode = 'Val'
        loss_window = []
        PA_window = []
        MA_window = []
        MI_window = []
        i = 0
        loader = tqdm(val_loader, desc='{}_Epoch:{:03d}'.format(mode, epoch), ncols=0)
        with torch.no_grad():
            for image, label in loader:
                if i >= CONFIG.SOLVER.VAL_IT:
                    break

                image = image.to(device=device)
                label = label.to(device=device)

                score = seg_model(image)
                loss1 = loss_layer(score[0], label)
                loss2 = loss_layer(score[1], label)
                loss3 = loss_layer(score[2], label)
                loss = loss1 + loss2 + loss3

                predict = score[2].detach().argmax(dim=1, keepdim=True).cpu().numpy()
                label = label.detach().cpu().numpy()
                ans = Model_Score(label, predict, CONFIG.DATASET.CLASS_NUM)
                loss_window.append(loss.detach().cpu())
                PA_window.append(ans["Pixel Accuracy"])
                MA_window.append(ans["Mean Accuracy"])
                MI_window.append[ans["Mean IoU"]]
                fmt = '{:.4f}'.format
                loader.set_postfix(loss=fmt(np.mean(loss_window)), pix_acc=fmt(np.mean(PA_window)))
                i += 1
        #Val

        name = "Epoch_{:d}".format(epoch)
        result = {
            'model':seg_model.state_dict(),
            'recorder':recoder
        }
        torch.save(results, os.path.join(CONFIG.SOLVER.SAVE_PATH, 'bisenet_{}.pth'.format(name)))

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

class BiSeNet(nn.Module):

    def __init__(self, class_nums):
        super().__init__()
        self.context_path = Backbone()
        self.spatial_path = SpatialPath(128)
        self.globalpooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv_ben_relu(512, 128, 1, 1, 0, bias=False)
        )
        
        ARMs = [AttentionRefinement(512, 128),
                AttentionRefinement(256, 128)]
        Refine = [conv_ben_relu(128, 128, 3, 1, 1, bias=False),
                conv_ben_relu(128, 128, 3, 1, 1, bias=False)] #keep size
        Scores = [Predict_Score(128, class_nums, 16),
                Predict_Score(128, class_nums, 8),
                Predict_Score(128 * 2, class_nums, 8)] #stage2,stage3,final

        self.arms = nn.ModuleList(ARMs)
        self.refines = nn.ModuleList(Refine)
        self.Scores = nn.ModuleList(Scores)
        self.ffm = FeatureFusionModule(128 * 2, 128 * 2, scale=4)

    def forward(self, x):
        spatial_feat = self.spatial_path(x)
        context_feat = self.context_path(x) #[stage3,stage4]
        global_context = self.globalpooling(context_feat[1])
        
        out = []
        feature = self.arms[0](context_feat[1]) + global_context
        feature = nn.functional.interpolate(feature, scale_factor=2, mode='bilinear', align_corners=True)
        feature = self.refines[0](feature)
        score = self.Scores[0](feature)
        out.append(score)

        feature = self.arms[1](context_feat[0]) + feature
        feature = nn.functional.interpolate(feature, scale_factor=2, mode='bilinear', align_corners=True)
        feature = self.refines[1](feature)
        score = self.Scores[1](feature)
        out.append(score)

        context_feat = feature

        final_feature = self.ffm(spatial_feat, context_feat)
        score = self.Scores[2](final_feature)
        out.append(score)

        return out

class FeatureFusionModule(nn.Module):
    
    def __init__(self, inplane, plane, scale=1):
        super().__init__()
        self.conv1 = conv_ben_relu(inplane, plane, 1, 1, 0, bias=False)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv_ben_relu(plane, plane // scale, 1, 1, 0, bias=False),
            conv_ben_relu(plane // scale, plane, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        att = self.attention(x)
        x = x + x * att

        return x

class Predict_Score(nn.Module):

    def __init__(self, inplane, plane, scale):
        super().__init__()
        self.conv1 = conv_ben_relu(inplane, 256, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(256, plane, 1, stride=1)
        self.scale = scale

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)

        return x

class AttentionRefinement(nn.Module):

    def __init__(self, inplane, plane):
        super().__init__()
        self.conv = conv_ben_relu(inplane, plane, 3, 1, 1, bias=False)#keep size
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv_ben_relu(plane, plane, 1, 1, 0, relu=False, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv(x)
        weight = self.attention()
        x = x * weight

        return x

class SpatialPath(nn.Module):

    def __init__(self, plane):
        super().__init__()
        inner_plane = 64
        self.conv1 = conv_ben_relu(3, inner_plane, 7, 2, 3, bias=False)
        self.conv2 = conv_ben_relu(inner_plane, inner_plane, 3, 2, 1, bias=False)
        self.conv3 = conv_ben_relu(inner_plane, inner_plane, 3, 2, 1, bias=False)
        self.conv4 = conv_ben_relu(inner_plane, plane, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        return x

class Backbone(nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()
        block = BasicBlock
        layers = [2, 2, 2, 2]
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        pre_model = resnet18(pretrained=True)
        pre_dict = pre_model.state_dict()
        self_dict = self.state_dict()

        pre_dict =  {k: v for k, v in pre_dict.items() if k in self_dict}
        self_dict.update(pre_dict)
        self.load_state_dict(self_dict)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        ans = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        ans.append(x) #256*28*28
        x = self.layer4(x)
        ans.append(x) #512*14*14

        return ans

def conv_ben_relu(in_planes, out_planes, kernal, stride, pad, bn=True, relu=True, bias=True):
    mlist = []
    mlist.append(nn.Conv2d(in_planes, out_planes, kernal, stride=stride, padding=pad, bias=bias))
    
    if bn:
        mlist.append(nn.BatchNorm2d(out_planes))

    if relu:
        mlist.append(nn.ReLU(inplace=True))

    return nn.Sequential(*mlist)    

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

if __name__ == "__main__":
    main()
