import torch.nn as nn
import math
import torch

class ResNetCifar_quantized(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.quant = torch.quantization.stubs.QuantStub()
        self.dequant = torch.quantization.stubs.DeQuantStub()
        quantization_config = torch.quantization.get_default_qconfig("fbgemm")
        self.qconfig = quantization_config
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

    def fuse_self(self):
        torch.quantization.fuse_modules(self.model, [["conv1", "bn1", "relu"]], inplace=True)
        for module_name, module in self.model.named_children():
            if "layer" in module_name:
                for basic_block_name, basic_block in module.named_children():
                    torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]],
                                                    inplace=True)
                    for sub_block_name, sub_block in basic_block.named_children():
                        if sub_block_name == "downsample":
                            torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

