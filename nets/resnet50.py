import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable


class Bottleneck(nn.Module):
    '''
    Contains three convolutional layers:
    conv1 - Reduces the number of channels
    conv2 - Extracts features
    conv3 - Expands the number of channels
    This structure helps better feature extraction, deepens the network, and reduces the number of parameters.
    '''

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        '''
        Implements the residual block structure.

        ResNet50 has two basic blocks: Conv Block and Identity Block.
        ResNet50 is built by stacking these two blocks.
        The main difference between them is whether the shortcut path has a convolution layer.

        Identity Block: Standard residual structure where the shortcut does not have a convolution;
        Conv Block: The shortcut includes a convolution and BN (Batch Normalization),
        which allows changing the dimensions of the network.

        That is:
        - Identity Block is used when input and output dimensions are the same, allowing deep stacking;
        - Conv Block is used to change dimensions and cannot be stacked consecutively.

        :param x: Input data
        :return: Output result from the network
        '''
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
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        # -----------------------------------#
        #   Assume input image is 600x600x3
        # -----------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()

        # 600x600x3 -> 300x300x64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 300x300x64 -> 150x150x64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150x150x64 -> 150x150x256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150x150x256 -> 75x75x512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75x75x512 -> 38x38x1024 (shared feature layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 is used in the classifier model
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        '''
        Builds stacked Conv Blocks and Identity Blocks
        :param block: The Bottleneck block defined above (basic unit in ResNet50)
        :param planes: Output channel count
        :param blocks: Number of repeated residual blocks
        :param stride: Convolution stride
        :return: Stacked Conv and Identity Block structure
        '''
        downsample = None
        # -------------------------------------------------------------------#
        #   When downsampling is needed (reduce H and W), use downsample in the shortcut path
        # -------------------------------------------------------------------#

        # Shortcut path (downsample for Conv Block)
        if stride != 1 or self.inplanes != planes * block.expansion:  # block.expansion = 4
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []  # Stack of Conv and Identity Blocks
        # Add a Conv Block
        layers.append(block(self.inplanes, planes, stride, downsample))
        # Update input dimensions after Conv Block
        self.inplanes = planes * block.expansion
        # Add remaining Identity Blocks
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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    # ----------------------------------------------------------------------------#
    #   Get feature extraction part: from conv1 to layer3, producing 38x38x1024 feature map
    # ----------------------------------------------------------------------------#
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # ----------------------------------------------------------------------------#
    #   Get classification part: from layer4 to avgpool
    # ----------------------------------------------------------------------------#
    classifier = list([model.layer4, model.avgpool])

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier

