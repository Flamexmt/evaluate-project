#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# The TorchVision implementation in https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# has 2 issues in the implementation of the BasicBlock and Bottleneck modules, which impact our ability to
# collect activation statistics and run quantization:
#   1. Re-used ReLU modules
#   2. Element-wise addition as a direct tensor operation
# Here we provide an implementation of both classes that fixes these issues, and we provide the same API to create
# ResNet and ResNeXt models as in the TorchVision implementation.
# We reuse the original implementation as much as possible.

from collections import OrderedDict
import torch.nn as nn
import torch
from distiller.modules import EltwiseAdd


__all__ = ['wide_resnet50_2_cifar']
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class DownSample(nn.Sequential):
    def __init__(self,in_channels, out_channels,
                          kernel_size=1, stride=0, bias=False):
        super(DownSample, self).__init__(
            nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class DistillerBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        # Initialize torchvision version
        super(DistillerBasicBlock, self).__init__(*args, **kwargs)

        # Remove original relu in favor of numbered modules
        delattr(self, 'relu')
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.add = EltwiseAdd(inplace=True)  # Replace '+=' operator with inplace module

        # Trick to make the modules accessible in their topological order
        modules = OrderedDict()
        modules['conv1'] = self.conv1
        modules['bn1'] = self.bn1
        modules['relu1'] = self.relu1
        modules['conv2'] = self.conv2
        modules['bn2'] = self.bn2
        if self.downsample is not None:
            modules['downsample'] = self.downsample
        modules['add'] = self.add
        modules['relu2'] = self.relu2
        self._modules = modules

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu2(out)

        return out


class DistillerBottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        # Initialize torchvision version
        super(DistillerBottleneck, self).__init__(*args, **kwargs)

        # Remove original relu in favor of numbered modules
        delattr(self, 'relu')
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.add = EltwiseAdd(inplace=True)  # Replace '+=' operator with inplace module

        # Trick to make the modules accessible in their topological order
        modules = OrderedDict()
        modules['conv1'] = self.conv1
        modules['bn1'] = self.bn1
        modules['relu1'] = self.relu1
        modules['conv2'] = self.conv2
        modules['bn2'] = self.bn2
        modules['relu2'] = self.relu2
        modules['conv3'] = self.conv3
        modules['bn3'] = self.bn3
        if self.downsample is not None:
            modules['downsample'] = self.downsample
        modules['add'] = self.add
        modules['relu3'] = self.relu3
        self._modules = modules

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu3(out)

        return out




class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.quant = torch.quantization.stubs.QuantStub()
        self.dequant = torch.quantization.stubs.DeQuantStub()
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    class DownSample(nn.Sequential):
        def __init__(self, in_channels, out_channels,
                     kernel_size=1, stride=0, bias=False):
            super(DownSample, self).__init__(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride),
            #     norm_layer(planes * block.expansion),
            # )
            downsample = DownSample(self.inplanes, planes * block.expansion,
                                    kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, ["conv1", "bn1", "relu"],inplace=True)
        for m in self.modules():
            if type(m) == BasicBlock:
                torch.quantization.fuse_modules(m,[['conv2','bn2'], ['conv1', 'bn1']] ,inplace=True)
            if type(m) == DownSample:
                torch.quantization.fuse_modules(m, ['0', '1'],inplace=True)
            if type(m) == DistillerBasicBlock:
                torch.quantization.fuse_modules(m,[['conv2','bn2'], ['conv1', 'bn1']] ,inplace=True)
            if type(m) == DistillerBottleneck:
                torch.quantization.fuse_modules(m, [['conv2', 'bn2'], ['conv1', 'bn1'],['conv3', 'bn3']], inplace=True)
        return self

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def wide_resnet50_2_cifar(pretrained=False, progress=True, **kwargs):
    """Constructs a Wide ResNet-50-2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 10
    return _resnet('wide_resnet50_2', DistillerBottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


