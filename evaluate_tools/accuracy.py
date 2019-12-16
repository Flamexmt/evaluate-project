import datetime
import os

import DataLoader
import  modelFactory
import torch
from torch.autograd import Variable

def test_accuracy(args):
    if args.dataset == 'cifar10':
        test_loader = _, DataLoader.cifar10_loader(args.dataset_path, args.batch_size)
    elif args.dataset=='minst':
        test_loader=_,DataLoader.mnist_loader(args.dataset_path,args.batch_size)
    elif args.dataset=='imagenet':
        print('imagenet')
    test(test_loader, args)


def test(test_loader,args):
    model=modelFactory.find_model(args.model)
    model.load_state_dict(torch.load(args.model_path))
    correct = 0  # 初始化预测正确的数据个数为0
    starttime = datetime.datetime.now()
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
    endtime=datetime.datetime.now()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
         correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('time cost is ',(endtime-starttime).seconds)
    print(os.path.getsize(args.model_path))#打印文件大小

