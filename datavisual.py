# Python实现正态分布
# 绘制正态分布概率密度函数
import math
import matplotlib.pyplot as plt
import numpy
import seaborn as sns
import torch
from scipy.stats import norm

model1 = torch.load('D:\study\model-compression\git-hub\evaluate-project\outputsdata/mobilenet_v2-b0353104.pth')
model2 = torch.load('D:\study\model-compression\git-hub\evaluate-project\outputsdata/resnet50-19c8e357.pth')
model3 = torch.load('D:\study\model-compression\git-hub\evaluate-project\outputsdata/wide_resnet50_2-95faca4d.pth')


def make_plot(model, pos=1, name=''):
    data_list = {}
    for k, v in model.items():
        if 'weight' in k and 'conv' in k:
            data_list[k] = v.detach().cpu().numpy().flatten()
    sorted_data_list = sorted(data_list.items(), key=lambda item: len(item[1]), reverse=True)
    i = 0
    for k, v in sorted_data_list:
        i += 1
        sns.kdeplot(data_list[k], label=k)
        if i > 9:
            break
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 25,
             }
    font2 = {
        'size': 28,
    }
    plt.tick_params(labelsize=20)
    legend = plt.legend(prop=font1)
    plt.xlabel('Value of Weight', font2)
    plt.ylabel('Density', font2)
    plt.xlim(-0.25, 0.25)

plt.figure(figsize=(12, 8))
make_plot(model3)
plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.12)
plt.show()
