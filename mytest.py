import torch

from cifar_models import resnet20_cifar
from simplenet_cifar import Simplenet, simplenet_cifar

model1=simplenet_cifar()
model2=resnet20_cifar()
print(model1)
print(model2)