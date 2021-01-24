import torch.nn as nn
import math
import torch
from distiller.models.cifar10.wideresnet import BasicBlock,DownSample,DistillerBasicBlock,DistillerBottleneck
class WideResNet_quantized(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.quant = torch.quantization.stubs.QuantStub()
        self.dequant = torch.quantization.stubs.DeQuantStub()
        quantization_config = torch.quantization.get_default_qconfig("fbgemm")
        self.qconfig = quantization_config
    def forward(self, x):
        x = x.to('cpu')
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

    def fuse_self(self):
        torch.quantization.fuse_modules(self.model, ["conv1", "bn1", "relu"], inplace=True)
        for m in self.modules():
            if type(m) == BasicBlock:
                torch.quantization.fuse_modules(m, [['conv2', 'bn2'], ['conv1', 'bn1']], inplace=True)
            if type(m) == DownSample:
                torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
            if type(m) == DistillerBasicBlock:
                torch.quantization.fuse_modules(m, [['conv2', 'bn2'], ['conv1', 'bn1']], inplace=True)
            if type(m) == DistillerBottleneck:
                torch.quantization.fuse_modules(m, [['conv2', 'bn2'], ['conv1', 'bn1'], ['conv3', 'bn3']], inplace=True)