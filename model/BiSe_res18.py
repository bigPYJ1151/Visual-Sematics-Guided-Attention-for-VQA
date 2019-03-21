
import torch
import torch.nn as nn
from torchvision.models import resnet18

class BiSeNet(nn.Module):

    def __init__(self, class_nums):
        super().__init__()
        self.context_path = Backbone()
        self.spatial_path = SpatialPath(128)
        self.globalpooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv_ben_relu(512, 128, 1, 1, 0, bias=False)
        )
        self.arms = nn.ModuleList([AttentionRefinement(512, 128),
                                    AttentionRefinement(256, 128)])
        self.refines = nn.ModuleList([conv_ben_relu(128, 128, 3, 1, 1, bias=False),
                                    conv_ben_relu(128, 128, 3, 1, 1, bias=False)])
        self.Scores = nn.ModuleList([Predict_Score(128, class_nums, 16),
                                     Predict_Score(128, class_nums, 8),
                                     Predict_Score(128 * 2, class_nums, 8)])
        self.ffm = FeatureFusionModule(128 * 2, 128 * 2, scale=4)
        #self.loss = nn.CrossEntropyLoss(ignore_index=ignored_label)

    def forward(self, x):
        spatial_feat = self.spatial_path(x)
        context_feat = self.context_path(x) #[stage3,stage4]
        global_context = self.globalpooling(context_feat[1])
        
        feature = self.arms[0](context_feat[1]) + global_context
        feature = nn.functional.interpolate(feature, scale_factor=2, mode='bilinear', align_corners=True)
        feature = self.refines[0](feature)
        score1 = self.Scores[0](feature)

        feature = self.arms[1](context_feat[0]) + feature
        feature = nn.functional.interpolate(feature, scale_factor=2, mode='bilinear', align_corners=True)
        feature = self.refines[1](feature)
        score2 = self.Scores[1](feature)

        context_feat = feature

        final_feature = self.ffm(spatial_feat, context_feat)
        score3 = self.Scores[2](final_feature)

        # loss1 = self.loss(score1, label)
        # loss2 = self.loss(score2, label)
        # loss3 = self.loss(score3, label)
        # loss = loss1 + loss2 + loss3

        return score1, score2, score3

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
        weight = self.attention(x)
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
