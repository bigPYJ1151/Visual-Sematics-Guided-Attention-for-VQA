
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNet152(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers = [3, 8, 36, 3]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(resnet.Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(resnet.Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(resnet.Bottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(resnet.Bottleneck, 512, layers[3], stride=2)

        pre_model = resnet.resnet152(pretrained=True)
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x