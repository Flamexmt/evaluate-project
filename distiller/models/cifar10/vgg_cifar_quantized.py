import torch.nn as nn
import math
import torch

class VggCifar_quantized(nn.Module):
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
        # torch.quantization.fuse_modules(self.model, [["conv1", "relu"]], inplace=True)
        pass


