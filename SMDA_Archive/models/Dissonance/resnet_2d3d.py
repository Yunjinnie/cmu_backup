## code source - https://github.com/TengdaHan/DPC/tree/master/backbone
## modified from https://github.com/kenshohara/3D-ResNets-PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Bottleneck2d(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_final_relu=True):
        super().__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=(1,stride,stride), padding=(0,1,1), bias=bias)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes*4, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes*4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.batchnorm = True
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batchnorm: out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batchnorm: out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.batchnorm: out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.use_final_relu:
            out = self.relu(out)
        return out

class Bottleneck3d(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_final_relu=True):
        super().__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes*4, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes*4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.use_final_relu:
            out = self.relu(out)
        return out

class ResNet2d3d_full(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        bias = False
        self.conv1 = nn.Conv3d(3,64,kernel_size=(1,7,7),stride=(1,2,2),padding=(0,3,3),bias=bias)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1))
        if not isinstance(block, list):
            block = [block] * 4
        
        self.layer1 = self._make_layer(block[0], 64, layers[0])
        self.layer2 = self._make_layer(block[1], 64*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block[2], 64*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block[3], 64*4, layers[3], stride=2, is_final=True)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1, is_final=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if block == Bottleneck2d: #or (block == BasicBlock2d):
                customized_stride = (1, stride, stride)
            else:
                customized_stride = stride
            
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=customized_stride, bias=False),
                nn.BatchNorm3d(planes*block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if is_final:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, use_final_relu=False))
        else:
            for i in range(1, blocks):
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


def resnet50_2d3d_full(**kwargs):
    model = ResNet2d3d_full(
        [Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d],
        [3,4,6,3],
        **kwargs)
    return model