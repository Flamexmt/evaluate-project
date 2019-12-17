import datetime
import os
import cifar_models.resnet as resnet
import DataLoader
import modelFactory
import torch
from torch.autograd import Variable

model=resnet.ResNet18()
dict=torch.load('D:\study\model-compression\git-hub\distiller\evaluate_tools\checkpoint.pt.best')
for key in dict.keys():
    print(key)
print(dict['model'].keys())
print(dict['model']['module.normalizer.new_mean'])
print(dict['model']['module.normalizer.new_std'])

print(type(dict['model']['module.attacker.normalize.new_mean']))

print(dict['model']['module.attacker.normalize.new_std'])

